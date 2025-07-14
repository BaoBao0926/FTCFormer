import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_utils import DropPath, to_2tuple, trunc_normal_
from .tcformer_utils import (
    merge_tokens, cluster_dpc_knn, token2map,
    map2token, token_downup, sra_flops, map2token_flops, token2map_flops, downup_flops, cluster_and_merge_flops)
import sys
import numpy as np
import time

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention module with spatial reduction layer
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Transformer blocks
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# The first conv layer
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# conv layer for dynamic tokens
class TokenConv(nn.Conv2d):
    def __init__(self, if_CTM=True, **kwargs):
        super().__init__(**kwargs)
        self.if_CTM=if_CTM
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, token_dict):
        x = token_dict['x']
        x = self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.if_CTM:
            x_map = token2map(token_dict)
        else:   
            B, N, C = x.shape
            H, W = token_dict['map_size']
            x_map = x.permute(0,2,1).view(B, C, H, W)   # [B,N,C]->[B,C,N]->[B,C,H,W]
        x_map = super().forward(x_map)
        if self.if_CTM:
            x = x + map2token(x_map, token_dict)
        else:
            x = x + x_map.view(B, C, H*W).permute(0,2,1)
        return x


# Mlp for dynamic tokens
class TCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., if_CTM=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenConv(in_channels=hidden_features,
                                out_channels=hidden_features,
                                kernel_size=3, padding=1, stride=1,
                                bias=True,
                                groups=hidden_features, if_CTM=if_CTM)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, token_dict):
        token_dict['x'] = self.fc1(token_dict['x'])
        x = self.dwconv(token_dict)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention for dynamic tokens
class TCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_sr_layer=True, if_CTM=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.if_CTM = if_CTM
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.use_sr_layer = use_sr_layer
        if sr_ratio > 1:
            if self.use_sr_layer:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q_dict, kv_dict):
        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1)

        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # print(f'kv {kv.shape}') kv torch.Size([60, 3136, 128])
            # print(f'conf_kv {conf_kv.shape}')   conf_kv torch.Size([60, 3136, 1])
            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            if self.if_CTM:
                tmp = token2map(tmp_dict)   # torch.Size([60, 129, 28, 28])
            else:
                B, N, C1 = tmp.shape
                H, W = kv_dict['map_size']
                tmp = tmp.permute(0,2,1).view(B, C1, H, W)   # torch.Size([60, 129, 28, 28])

            kv = tmp[:, :C]
            conf_kv = tmp[:, C:]

            if self.use_sr_layer:
                # print(f'kv {kv.shape}') # kv torch.Size([60, 129, 28, 28])  kv torch.Size([60, 128, 28, 28])
                kv = self.sr(kv)    # kv torch.Size([60, 128, 28, 28])
                _, _, h, w = kv.shape
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                kv = self.norm(kv)
            else:
                kv = F.avg_pool2d(kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            conf_kv = F.avg_pool2d(conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer block for dynamic tokens
class TCBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=True,
                 if_CTM=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TCAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_sr_layer=use_sr_layer, if_CTM=if_CTM)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, if_CTM=if_CTM)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None

        x = q_dict['x']

        # norm1
        q_dict['x'] = self.norm1(q_dict['x'])
        if kv_dict is None:
            kv_dict = q_dict
        else:
            kv_dict['x'] = self.norm1(kv_dict['x'])
        # torch.Size([60, 784, 128])        torch.Size([60, 784, 128])
        # torch.Size([60, 784, 128])        torch.Size([60, 3136, 128])
        # attn
        x = x + self.drop_path(self.attn(q_dict, kv_dict))

        # mlp
        q_dict['x'] = self.norm2(x)
        x = x + self.drop_path(self.mlp(q_dict))
        q_dict['x'] = x

        return q_dict

# CTM block
class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5, k_WSN=5, Cmerge=True, FDPC_KNN=True, if_WSN=True):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.k = k
        # new hyper-parameter
        self.Cmerge = Cmerge
        self.FDPC_KNN = FDPC_KNN
        self.if_WSN = if_WSN
        self.k_WSN = k_WSN
        if Cmerge:
            self.score = nn.Linear(self.dim_out, self.dim_out)
        else:
            self.score = nn.Linear(self.dim_out, 1)

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        H, W = token_dict['map_size']
        x = self.conv(token_dict)
        x = self.norm(x)
        B, N, C = x.shape


        token_score = self.score(x)
        # log-sum-exp trick
        max_scores = token_score.max(dim=1, keepdim=True).values  # [batch, 1, channels]
        token_weight = (token_score-max_scores).exp()
        token_dict['x'] = x
        if self.Cmerge:
            token_dict['token_score'] = token_score.mean(-1, keepdim=True)
        else:
            token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)

        if self.FDPC_KNN and self.if_WSN:       # fuzzy DPC-KNN + WSN
            idx_cluster, cluster_num = self.cluster_dpc_fknn_WSN(token_dict, cluster_num, self.k, self.k_WSN)
        elif self.FDPC_KNN and not self.if_WSN: # fuzzy DPC-KNN, not WSN
            idx_cluster, cluster_num = self.cluster_dpc_fknn(token_dict, cluster_num, self.k)
        elif not self.FDPC_KNN and self.if_WSN:     # not fuzzy DPC-KNN, WSN
            idx_cluster, cluster_num = self.cluster_dpc_knn_WSN(token_dict, cluster_num, self.k, self.k_WSN)
        elif not self.FDPC_KNN and not self.if_WSN:   # not fuzzy DPC-KNN, not WSN
            idx_cluster, cluster_num = self.cluster_dpc_knn(token_dict, cluster_num, self.k) 
        else:
            raise RuntimeError(f'please check FDPC_KNN(value: {self.FDPC_KNN}) and if_WSN(value: {self.if_WSN})')

        if self.Cmerge:
            down_dict = self.Cmerge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        else:
            down_dict = self.merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)



        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict

    # ---------------------- token merging
    def merge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight=None):
        """Merge tokens in the same cluster to a single cluster.
        Implemented by torch.index_add(). Flops: B*N*(C+2)
        Return:
            out_dict (dict): dict for output token information

        Args:
            token_dict (dict): dict for input token information
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): cluster number
            token_weight (Tensor[B, N, 1]): weight for each token.
        """

        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num
        
        all_weight = token_weight.new_zeros(B * cluster_num, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                            source=token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]        # torch.Size([60, 3136, 1])

        # average token features
        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        idx_token_new = self.index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight, idx_token)
        agg_weight_new = agg_weight * weight_t
        agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = cluster_num
        out_dict['map_size'] = token_dict['map_size']
        out_dict['init_grid_size'] = token_dict['init_grid_size']
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = agg_weight_new
        return out_dict

    def Cmerge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight=None):
        """Merge tokens in the same cluster to a single cluster.
        Implemented by torch.index_add(). Flops: B*N*(C+2)
        Return:
            out_dict (dict): dict for output token information

        Args:
            token_dict (dict): dict for input token information
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): cluster number
            token_weight (Tensor[B, N, 1]): weight for each token.
        """

        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, C)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num

        # token weight + C-weight
        all_weight = token_weight.new_zeros(B * cluster_num, C)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, C))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # print(norm_weight.shape)
        token_weight_for_norm = token_weight.mean(-1, keepdim=True)
        all_weight_token = token_weight_for_norm.new_zeros(B * cluster_num, 1)
        all_weight_token.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight_for_norm.reshape(B * N, 1))
        all_weight_token = all_weight_token + 1e-6
        norm_weight_token = token_weight_for_norm / all_weight_token[idx]

        # average token features
        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        idx_token_new = self.index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight_token, idx_token)
        # print(agg_weight.shape)
        # print(weight_t.shape)

        agg_weight_new = agg_weight * weight_t
        agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = cluster_num
        out_dict['map_size'] = token_dict['map_size']
        out_dict['init_grid_size'] = token_dict['init_grid_size']
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = agg_weight_new
        return out_dict

    # ------------------------------ DPC-KNN ------------------
    def cluster_dpc_knn(self, token_dict, cluster_num, k=5, token_mask=None):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
                total time 0.12060689926147461  local density 0.0010237693786621094  t2 0.1197960376739502
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)        # 0.119
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    def cluster_dpc_knn_WSN(self, token_dict, cluster_num, k=5, k_WSN=5, token_mask=None):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
                total time 0.12060689926147461  local density 0.0010237693786621094  t2 0.1197960376739502
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # calculate WSN and feature distance
            _, nn_indices = torch.topk(-dist_matrix, k=k_WSN, dim=-1)  # Nearest neighbors
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)

            feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
            WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            final_score = feature_distance_score + 100000*WSN_score
            # assign
            idx_cluster = final_score.argmax(dim=1)       # torch.Size([10, 128, 7680])

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    def cluster_dpc_fknn(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Fuzzy DPC-KNN
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Assign tokens to nearest cluster centers
            dist_matrix = self.index_points(dist_matrix, index_down)        # smaller here means closer
            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    def cluster_dpc_fknn_WSN(self, token_dict, cluster_num, k=5, k_WSN=5, token_mask=None):
        """
        Fuzzy DPC-KNN + WSN
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            # nn_mask[batch_indices, token_indices, nn_indices] = True      # 不需要这个nn_mask可以省内存

            # u
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            mu = (torch.exp(- (dist_selected)**2)/ (dist_selected + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = mu.sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            # rho_non_nn = ((mu * (~nn_mask)).sum(dim=-1) - rho_nn) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho_non_nn = mu_non_nn.sum(-1) / N      # 不是按照原来的公式，但是可以省内存
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        
            _, nn_indices = torch.topk(-dist_matrix, k=k_WSN, dim=-1)  # Nearest neighbors
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
            final_score = feature_distance_score + 100000*WSN_score

            idx_cluster = final_score.argmax(dim=1)       # torch.Size([10, 128, 7680])
            # make sure cluster center merge to itself  
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)    # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        
        return idx_cluster, cluster_num

    # ------------------------------help function ------------------------
    def index_points(self, points, idx):
        """Sample features following the index.
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def get_neighbor_mask(self, h, w, k):
        """
        生成二维图片每个token的周围k个邻居的掩码矩阵，包括对角线的邻居。
        Args:
            h: 图片的高度
            w: 图片的宽度
            k: 邻居的半径（欧几里得距离，包含对角线）

        Returns:
            neighbor_mask: Tensor [N, N], 每个位置表示是否是邻居
        """
        N = h * w
        coords = torch.arange(N).view(h, w)  # 生成二维坐标索引
        # coords_flat = coords.flatten()  # 展平为一维

        # 获取二维坐标
        x = torch.arange(h).view(-1, 1).repeat(1, w).flatten()
        y = torch.arange(w).repeat(h).flatten()

        # 计算所有点之间的欧几里得距离
        euclidean_dist = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2 + (y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sqrt()

        # 生成邻居掩码矩阵
        neighbor_mask = euclidean_dist <= k
        return neighbor_mask

    def calculate_WSN(self, dist_selected, dist_matrix, nn_indices, index_down):
        W = (1 / (dist_selected + 1)).sum(dim=-1)  # Shape: (B, N)  把单独能的，换成上面用过的dist_selected，即只有knn的距离
        W = W.unsqueeze(-1) + W.unsqueeze(-2)  # [优化] 避免 `W[:, :, None] + W[:, None, :]` 造成大规模广播
    
        knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)  # in-place scatter operation
        # Calculate shared neighbors in-place to save memory
        snn = knn_mask @ knn_mask.transpose(1, 2)

        WSN = snn * W
        WSN = self.index_points(WSN, index_down)
        return WSN
    
    def calculate_feature_distance_score(self, dist_matrix, index_down):
        dist_matrix = self.index_points(dist_matrix, index_down)
        return -dist_matrix



class Conv_for_dict(nn.Module):
    def __init__(self, in_channel, out_channel, downsampling='Convk33s2'):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=1,stride=1,padding=0)
        # self.score = nn.Linear(in_channel, 1)
        if downsampling == 'Convk33s2':
            self.downsampling = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3,stride=2,padding=1)
        elif downsampling == 'Convk22s2':
            self.downsampling = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=2,stride=2,padding=0)
        elif downsampling == 'maxpooling':
            self.downsampling = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=1,stride=1,padding=1)
            )
        elif downsampling ==  'avgpooling':
            self.downsampling = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=1,stride=1,padding=1)
            )
        else:
            raise RuntimeError('downsampling should be CTM, Convk33s2, Convk22s2, maxpooling')
        
    def forward(self, token_dict):
        x = token_dict['x']     # torch.Size([60, 3136, 64])
        H, W = token_dict['map_size']
        B, N, C = x.shape
        # token_dict['x'] = self.conv(x.view(B, H, W, C).permute(0,3,1,2)).permute(0,2,3,1).view(B, N,-1)
        # token_score = self.score(x)
        # token_dict['token_score'] = token_score
        x = self.downsampling(x.view(B, H, W, C).permute(0,3,1,2)).permute(0,2,3,1)
        B, H, W, C = x.shape
        N = H*W
        x = x.view(B, N, C)
        
        down_dict = {}
        down_dict['x'] = x
        down_dict['token_num'] = N
        down_dict['map_size'] = [H, W]
        down_dict['init_grid_size'] = token_dict['init_grid_size']
        down_dict['idx_token'] = torch.arange(N*4)[None, :].repeat(B, 1).to(x.device)
        down_dict['agg_weight'] = x.new_ones(B, N, 1)


        return down_dict #, token_dict

# CTM block
class CTM_backup1(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5,
                 # ------------------------ local density ---Token Selection --------------------------
                 if_local_density=True, k_local=1.5,
                 # ------------------------ Spatial Excitation ---Token Selection --------------------------
                 if_spatial_excitation=False, alpha=0.8, if_ex_density=True, if_ex_distance=True,
                 # ------------------------- MCTA ---------Token Assignment ------------------------
                 if_MCTA=False, if_cosine_similarity=True, if_attention_score=True, if_distance_score=True, if_feature_distance_scroe=True, if_KL_score=True, if_SSIM_score=True,
                 if_auto_weight=True, weight=[0.25,0.25,0.25,0.25],
                 if_fuzzy_merge=False, k_merge=3,
                 mode=1,
                 window_size=4,
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k
        # ------------------------ local density ---Token Selection --------------------------
        self.if_local_density = if_local_density
        self.k_local = k_local
        # ------------------------ Spatial Excitation ---Token Selection --------------------------
        self.if_spatial_excitation=if_spatial_excitation
        self.alpha = alpha
        self.if_ex_density=if_ex_density
        self.if_ex_distance=if_ex_distance
        # ------------------------- MCTA ---------Token Assignment ------------------------
        self.if_MCTA = if_MCTA
        self.if_cosine_similarity = if_cosine_similarity
        self.if_attention_score = if_attention_score
        self.if_feature_distance_scroe = if_feature_distance_scroe
        self.if_KL_score = if_KL_score
        self.if_SSIM_score = if_SSIM_score
        # ------------------------ fuzzy merge -----------------------
        self.if_fuzzy_merge = if_fuzzy_merge
        self.k_merge = k_merge
        # if if_attention_score:
        #     self.projection = nn.Linear(dim_out, dim_out)
        # else:
        #     self.projection = None
        self.window_size=window_size
        self.mode = mode
        

        self.if_auto_weight = if_auto_weight
        if if_auto_weight:
            raise RuntimeError('learnable weight for metrics do not backward update due to unknow reason')
            # n = sum([self.if_MCTA, self.if_cosine_similarity, self.if_attention_score, self.if_feature_distance_scroe])
            self.cos_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.att_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.spa_dis_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.fea_dis_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            # self.weight = [self.cos_weight, self.att_weight, self.spa_dis_weight, self.fea_dis_weight]
            # weight = nn.Parameter(torch.ones(4))
            # self.weight = nn.ParameterList([
            #     nn.Parameter(torch.tensor(1.0), requires_grad=True),    # cosine similarity
            #     nn.Parameter(torch.tensor(1.0), requires_grad=True),    # attention score
            #     nn.Parameter(torch.tensor(1.0), requires_grad=True),    # spatial distance
            #     nn.Parameter(torch.tensor(1.0), requires_grad=True)     # feature distance
            # ])
        else:
            self.weight = weight
        self.if_distance_score = if_distance_score

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        H, W = token_dict['map_size']
        x = self.conv(token_dict)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()
        B, N, C = x.shape

        if self.mode == 6:      # 这里是做tcformerv2的操作
            # 这里修改一下大小
            x = x.view(B,H,W,C)
            # 2. 划分成窗口 [B, H/a, a, W/a, a, C]
            x = x.view(B, H//self.window_size, self.window_size, W//self.window_size, self.window_size, C)  # [B, H/a, a, W/a, a, C]
            # 3. 调整维度顺序，使 batch 维度扩展
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, a, a, H/a, W/a, C]
            x = x.view(B*self.window_size*self.window_size, H//self.window_size, W//self.window_size, C)  # [B*a*a, H/a, W/a, C]

        token_dict['x'] = x
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)

        if self.mode == 1:
            # print(f'make sure coming here')
            idx_cluster, cluster_num = self.snn_dpc(token_dict, cluster_num)
        elif self.mode == 2:        # stadard varation weighted feauture
            idx_cluster, cluster_num = self.cluster_dpc_knn_StdVar(token_dict, cluster_num, self.k) 
        elif self.mode == 3:        # FKNN
            idx_cluster, cluster_num = self.cluster_dpc_fknn(token_dict, cluster_num, self.k) 
        elif self.mode == 4:        # FKNN+MCTA
            idx_cluster, cluster_num = self.cluster_dpc_fknn_MCTA(token_dict, cluster_num, self.k, H=H, W=W)
        elif self.mode == 5:        # FKNN+StdVar
            idx_cluster, cluster_num = self.cluster_dpc_fknn_StdVar(token_dict, cluster_num, self.k)
        elif self.mode == 6:        # TCFormerv2 还有问题暂时
            idx_cluster, cluster_num = self.cluster_dpc_knnv2(token_dict, cluster_num, self.k)
        elif self.mode == 7:        # FKNN + Excitation
            idx_cluster, cluster_num = self.cluster_dpc_fknn_excitation(token_dict, cluster_num, self.k)
        elif self.mode == 8:
            idx_cluster, cluster_num = self.cluster_dpc_fknn_wsn_simplify_v1(token_dict, cluster_num, self.k)
        elif self.mode == 9:
            idx_cluster, cluster_num = self.cluster_dpc_fknn_wsn_simplify_v2(token_dict, cluster_num, self.k)
        elif self.mode == 10:
            idx_cluster, cluster_num = self.cluster_dpc_fknnv2_WSN_MCTA(token_dict, cluster_num, self.k, H=H, W=W)
        elif self.mode == 11:
            idx_cluster, cluster_num = self.cluster_dpc_fknnv2_WSN_MCTA_seq(token_dict, cluster_num, self.k, H=H, W=W)
        elif self.mode == 12:
            idx_cluster, cluster_num = self.cluster_dpc_fknnv2_WSNseq_FMerge(token_dict, cluster_num, self.k, H=H, W=W)
        elif self.if_MCTA or self.if_spatial_excitation:
            idx_cluster, cluster_num = self.cluster_dpc_knn_MCTA(token_dict, cluster_num, self.k, H=H, W=W)
        else:
            idx_cluster, cluster_num = self.cluster_dpc_knn(token_dict, cluster_num, self.k) 
        
        if self.if_fuzzy_merge or self.mode == 12:
            down_dict = self.fuzzy_merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        else:
            down_dict = self.merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        # 恢复
        if self.mode == 6:
            x = down_dict['x'].view(B, self.window_size, self.window_size, H//self.window_size, W//self.window_size, C)  # [B, a, a, H/a, W/a, C]
            # 2. 调整维度顺序，使其回到 [B, H/a, a, W/a, a, C]
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, H/a, a, W/a, a, C]
            # 3. 重新 reshape 回到 [B, H, W, C]
            x = x.view(B, H, W, C)  # [B, H, W, C]
            # 4. 重新 reshape 到 [B, N, C]，其中 N = H*W
            down_dict['x'] = x.view(B, H * W, C)  # [B, N, C]


        return down_dict, token_dict

    # ---------------------- token merging
    def merge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight=None):
        """Merge tokens in the same cluster to a single cluster.
        Implemented by torch.index_add(). Flops: B*N*(C+2)
        Return:
            out_dict (dict): dict for output token information

        Args:
            token_dict (dict): dict for input token information
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): cluster number
            token_weight (Tensor[B, N, 1]): weight for each token.
        """

        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num

        all_weight = token_weight.new_zeros(B * cluster_num, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                            source=token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                            source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        idx_token_new = self.index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight, idx_token)
        agg_weight_new = agg_weight * weight_t
        agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = cluster_num
        out_dict['map_size'] = token_dict['map_size']
        out_dict['init_grid_size'] = token_dict['init_grid_size']
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = agg_weight_new
        return out_dict

    def fuzzy_merge_tokens(self, token_dict, idx_cluster, cluster_num, token_weight=None):
        """Merge tokens in the same cluster to a single cluster.
        Implemented by torch.index_add(). Flops: B*N*(C+2)
        Return:
            out_dict (dict): dict for output token information

        Args:
            token_dict (dict): dict for input token information
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): cluster number
            token_weight (Tensor[B, N, 1]): weight for each token.
        """

        x = token_dict['x']
        idx_token = token_dict['idx_token']
        agg_weight = token_dict['agg_weight']

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]       # tensor([[0]]), Batch_size = 1就是这样子的
        # tensor([[[0, 0, 0, 0, 1, 1, 1, 2, 2]]])， 由于Bs=1，所以index就是index，这里的做法是，把bs与N变成一个维度，然后乘上去 
        # torch.Size([1, 2, 9]
        idx_total = idx_cluster + idx_batch * cluster_num                 


        x_merged = x.new_zeros(B * cluster_num, C)
        norm = 0
        norm_weight_for_stroe = 0

        for i in range(self.k_merge):
            # print(f'---------------------- {i}-th --------------------------')
            idx = idx_total[:, i, :]
            all_weight = token_weight.new_zeros(B * cluster_num, 1) # tensor([[0.], [0.], [0.]])
            all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))   
            all_weight = all_weight + 1e-6
            norm_weight = token_weight / all_weight[idx]
            if i == 0: 
                norm_weight_for_stroe = norm_weight

            # average token features
            x_merged_temp = x.new_zeros(B * cluster_num, C)
            source = x * norm_weight / ((i+1)*(i+1))
            x_merged_temp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
            x_merged = x_merged + x_merged_temp
            norm = norm + 1/((i+1)*(i+1))

        x_merged = x_merged.reshape(B, cluster_num, C) / norm
        idx_token_new = self.index_points(idx_cluster[:, 0, :].view(B, N)[..., None], idx_token).squeeze(-1)
        weight_t = self.index_points(norm_weight_for_stroe, idx_token)
        agg_weight_new = agg_weight * weight_t
        agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        out_dict = {}
        out_dict['x'] = x_merged
        out_dict['token_num'] = cluster_num
        out_dict['map_size'] = token_dict['map_size']
        out_dict['init_grid_size'] = token_dict['init_grid_size']
        out_dict['cluster_idx_for_now'] = idx_cluster
        out_dict['idx_token'] = idx_token_new
        out_dict['agg_weight'] = agg_weight_new
        return out_dict

    # ----------------------指标计算
    def calculate_cosine_similarity(self, x, index_down):    
        """
        x: [B,N,C] torch.Size([10, 7680, 256]) 
        """
        # Step 1: 对最后一个维度进行 L2 归一化
        x_normalized = F.normalize(x, p=2, dim=-1)  # 形状仍为 [B, N, C] torch.Size([10, 7680, 256]) 
        # print(x_normalized.shape)
        # Step 2: 计算 token 间的余弦相似度, [B,N,N]： (i,j) is the cosine similarity between i-th token and j-th token
        cosine_score  = torch.matmul(x_normalized, x_normalized.transpose(1, 2))  # [B, N, N]    torch.Size([10, 7680, 7680])
        cosine_score = self.index_points(cosine_score, index_down)  # cosine_matrix is torch.Size([10, 128, 7680])
        return cosine_score

    def calculate_attention_score(self, x, index_down, r=1):
        """
        x: [B,N,C] torch.Size([10, 7680, 256]) 
        """
        # x_proj = attn_proj_layer(x)         # x_proj.shape torch.Size([10, 7680, 256])
        attn_matrix = torch.matmul(x, x.transpose(1,2))/r   # attn_matrix.shape torch.Size([10, 7680, 7680])
        attn_score = F.softmax(attn_matrix, dim=-1)      # torch.Size([10, 7680, 7680]) 
        attn_score = self.index_points(attn_score, index_down)   # attn_score.shape torch.Size([10, 128, 7680])
        return  attn_score

    def calculate_distance_score(self, x, index_down, H, W, distance_const=2):
        """
        Calculate the distance score tensor for tokens based on their spatial positions in the image.

        Args:
            x (torch.Tensor): Tensor of shape [B, N, C], where N = H * W.
            index_down: (not used in the current function, kept for compatibility with input format).
            H (int): Height of the original image.
            W (int): Width of the original image.

        Returns:
            torch.Tensor: [B, N/sample rate, N], which is spatial distance score
        """
        B, N, C = x.shape

        # Generate the grid of positions for the original image
        positions = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device), 
            torch.arange(W, device=x.device),
            # indexing="ij"     # old torch version do not have this parameter
        ), dim=-1)  # Shape: [H, W, 2]
        # Flatten the positions to match the tokens in x
        positions = positions.view(-1, 2)  # Shape: [N, 2], where N = H * W
        # Compute pairwise differences between positions
        diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # Shape: [N, N, 2]
        # Compute the Euclidean distance
        distances = torch.sqrt((diffs ** 2).sum(dim=-1))  # Shape: [N, N]
        # Expand distances for each batch
        distance_tensor = distances.unsqueeze(0).repeat(B, 1, 1)  # Shape: [B, N, N]

        # distance_score = 1 / (self.index_points(distance_tensor, index_down) + 1e-8)   # shape torch.Size([10, 128, 7680])
        distance_score = 1 / (self.index_points(distance_tensor, index_down)**2+ distance_const)   # shape torch.Size([10, 128, 7680])
        
        return distance_score

    def calculate_feature_distance_score(self, dist_matrix, index_down):
        dist_matrix = self.index_points(dist_matrix, index_down)
        return -dist_matrix

    def calculate_spatial_excitation(self, dist_matrix, k, alpha):
        """
        Optimized version of the spatial excitation calculation to reduce memory usage.

        Args:
            dist_matrix (Tensor[B, N, N]): Pairwise distance matrix.
            index_nearest (Tensor[B, N, k]): Indices of the k-nearest neighbors for each token.
            k (int): Number of nearest neighbors to consider.
            alpha (float): Balancing parameter for rank-based and distance weights.

        Returns:
            Tensor[B, N, k]: Spatial excitation values for the k-nearest neighbors.
        """
        # B, N, _ = dist_matrix.shape

        # # Gather the distances of the k-nearest neighbors directly
        # dist_nearest = torch.gather(dist_matrix, -1, index_nearest)  # Shape: [B, N, k]
        # Calculate rank-based weights for k-nearest neighbors
        rank_based_weights = (1 - alpha) * (torch.arange(1, k + 1, device=dist_matrix.device).float() / k) + alpha
        rank_based_weights = rank_based_weights.unsqueeze(0) # .unsqueeze(0)  # Shape: [1, 1, k]
        # # Multiply distances by rank-based weights
        # spatial_excitation = dist_nearest * rank_based_weights  # Shape: [B, N, k]

        return rank_based_weights

    def calculate_mahalanobis_scorev1(self, tokens, index_down):
        """
        计算每个 batch 内所有 token 之间的马氏距离矩阵 (无 for 循环)。

        :param tokens: [B, N, C] 的张量，表示 B 个 batch，每个 batch 有 N 个 tokens，每个 token C 维特征
        :return: [B, N, N] 的马氏距离矩阵，(i, j) 处表示 token i 到 token j 的马氏距离
        """
        B, N, C = tokens.shape

        # 计算协方差矩阵 S 并求逆
        tokens_centered = tokens - tokens.mean(dim=1, keepdim=True)  # [B, N, C]
        cov_matrices = torch.matmul(tokens_centered.transpose(-1, -2), tokens_centered) / (N - 1)  # [B, C, C]

        # 计算协方差矩阵的伪逆（避免不可逆问题）+ 处理 float16
        cov_inv = torch.linalg.pinv(cov_matrices.to(torch.float32)).to(tokens.dtype)  # [B, C, C]

        # 计算 tokens @ cov_inv.T，减少后续计算
        cov_inv_tokens = torch.matmul(tokens, cov_inv)  # [B, N, C]

        # 计算马氏距离公式： d_M(x_i, x_j) = sqrt((x_i - x_j)ᵀ S⁻¹ (x_i - x_j))
        # 直接计算 d_M^2(x_i, x_j) = (x_i - x_j)ᵀ S⁻¹ (x_i - x_j) 避免构造大矩阵
        mahalanobis_sq = (
            torch.sum(tokens * cov_inv_tokens, dim=-1, keepdim=True)  # [B, N, 1]
            - 2 * torch.matmul(cov_inv_tokens, tokens.transpose(-1, -2))  # [B, N, N]
            + torch.sum(tokens * cov_inv_tokens, dim=-1, keepdim=True).transpose(-1, -2)  # [B, 1, N]
        )

        mahalanobis_matrix = torch.sqrt(torch.clamp(mahalanobis_sq, min=0))  # 取 sqrt 计算马氏距离

        return -self.index_points(mahalanobis_matrix, index_down)

    def calculate_mahalanobis_scorev2(self, tokens, index_down, rank=64):
        """
        计算每个 batch 内所有 token 之间的马氏距离矩阵 (无 for 循环)，使用 SVD 近似求逆优化计算。

        :param tokens: [B, N, C] 的张量，表示 B 个 batch，每个 batch 有 N 个 tokens，每个 token C 维特征
        :param index_down: 索引，用于 index_points 操作
        :param rank: SVD 近似求逆的秩，默认值为 64
        :return: [B, N, N] 的马氏距离矩阵，(i, j) 处表示 token i 到 token j 的马氏距离
        """
        B, N, C = tokens.shape

        # 计算协方差矩阵 S
        tokens_centered = tokens - tokens.mean(dim=1, keepdim=True)  # [B, N, C]
        cov_matrices = torch.matmul(tokens_centered.transpose(-1, -2), tokens_centered) / (N - 1)  # [B, C, C]

        # **使用 SVD 低秩近似求逆**
        U, S, Vh = torch.linalg.svd(cov_matrices.to(torch.float32), full_matrices=False)  # SVD 分解
        S_inv = torch.diag_embed(1.0 / (S[:, :rank] + 1e-6))  # 计算前 rank 个特征值的逆
        cov_inv = (Vh[:, :rank, :].transpose(-1, -2) @ S_inv @ U[:, :, :rank].transpose(-1, -2)).to(tokens.dtype)  # 近似逆

        # 计算 tokens @ cov_inv.T，减少后续计算
        cov_inv_tokens = torch.matmul(tokens, cov_inv)  # [B, N, C]

        # 计算马氏距离公式： d_M(x_i, x_j) = sqrt((x_i - x_j)ᵀ S⁻¹ (x_i - x_j))
        mahalanobis_sq = (
            torch.sum(tokens * cov_inv_tokens, dim=-1, keepdim=True)  # [B, N, 1]
            - 2 * torch.matmul(cov_inv_tokens, tokens.transpose(-1, -2))  # [B, N, N]
            + torch.sum(tokens * cov_inv_tokens, dim=-1, keepdim=True).transpose(-1, -2)  # [B, 1, N]
        )

        mahalanobis_matrix = torch.sqrt(torch.clamp(mahalanobis_sq, min=0))  # 取 sqrt 计算马氏距离

        return -self.index_points(mahalanobis_matrix, index_down)

    def calculate_mahalanobis_score(self, tokens, index_down, rank=64):
        B, N, C = tokens.shape

        # 计算协方差矩阵 S
        tokens_centered = tokens - tokens.mean(dim=1, keepdim=True)  # [B, N, C]
        cov_matrices = torch.matmul(tokens_centered.transpose(-1, -2), tokens_centered) / (N - 1)  # [B, C, C]

        # 使用 SVD 低秩近似求逆
        U, S, Vh = torch.linalg.svd(cov_matrices.to(torch.float32), full_matrices=False)
        S_inv = torch.diag_embed(1.0 / (S[:, :rank] + 1e-6))  # 前 rank 个特征值的逆
        cov_inv = (Vh[:, :rank, :].transpose(-1, -2) @ S_inv @ U[:, :, :rank].transpose(-1, -2)).to(tokens.dtype)  # 近似逆

        # 计算变换后的特征并选择索引对应的行
        cov_inv_tokens = torch.matmul(tokens, cov_inv)  # [B, N, C]
        # cov_inv_tokens_selected = self.index_points(cov_inv_tokens, index_down)  # [B, K, C]

        # 提前选择需要计算的索引对应的特征
        # tokens_selected = self.index_points(tokens, index_down)  # [B, K, C]

        # 计算各项的选取部分
        sum1 = torch.sum(tokens * cov_inv_tokens, dim=-1, keepdim=True)  # [B, N, 1]
        # sum1_selected = self.index_points(sum1, index_down)  # [B, K, 1]
        sum2_transposed = sum1.transpose(-1, -2)  # [B, 1, N]

        # 仅计算索引对应的中间矩阵部分
        intermediate = torch.matmul(cov_inv_tokens, tokens.transpose(-1, -2))  # [B, K, N]
        mahalanobis_sq = (
            sum1
            - 2 * intermediate 
            + sum2_transposed
        )

        # 计算马氏距离并返回选取部分
        mahalanobis_matrix = torch.sqrt(torch.clamp(mahalanobis_sq, min=0))
        
        return -self.index_points(mahalanobis_matrix, index_down)

    def calculate_KL_score(self, tokens, index_down):
        B, N, C = tokens.shape
        epsilon = 1e-6

        # 归一化为概率分布
        P = F.softmax(tokens, dim=-1)  # [B, N, C]
        logP = torch.log(P + epsilon)  # 防止log(0)

        # 计算每个token的熵 H = Σ(P_i * logP_i), [B, N]
        H = (P * logP).sum(dim=-1)

        # 计算交叉熵矩阵 CE_ij = Σ(P_i *logQ_j) = Σ(P_i* logP_j)
        CE = torch.matmul(P, logP.transpose(-1, -2))  # [B, N, N]

        # KL散度矩阵 = H[:, None] - CE
        kl_div = H.unsqueeze(-1) - CE  # [B, N, N]
        return -self.index_points(kl_div, index_down)

    def calculate_SSIM_score(self, x, index_down, k1=0.01, k2=0.03):
        """
        x形状: [B, N, C]
        返回形状: [B, N, N] 的SSIM矩阵
        """
        B, N, C = x.shape
        L = 1  # 假设数据已归一化到[0,1]
        c1 = (k1 *L)**2
        c2 = (k2 *L)**2
        
        # 计算均值
        mu = x.mean(dim=-1, keepdim=True)  # [B, N, 1]
        
        # 计算协方差矩阵
        x_centered = x - mu
        sigma_xy = torch.einsum('bnc,bmc->bnm', x_centered, x_centered) / (C - 1)  # [B, N, N]
        
        # 提取对角线作为方差（σ_x²和σ_y²）
        sigma_xx = torch.diagonal(sigma_xy, dim1=1, dim2=2).unsqueeze(-1)  # [B, N, 1]
        sigma_yy = sigma_xx.transpose(1, 2)  # [B, 1, N]
        
        # 广播计算SSIM分子分母
        mu1 = mu  # [B, N, 1]
        mu2 = mu.transpose(1, 2)  # [B, 1, N]
        
        numerator = (2 *(mu1* mu2) + c1) *(2* sigma_xy + c2)  # 广播到[B, N, N]
        denominator = (mu1**2 + mu2**2 + c1)* (sigma_xx + sigma_yy + c2)
        
        ssim_matrix = numerator / (denominator + 1e-8)  # 防止除以零
            
        return self.index_points(ssim_matrix, index_down) # [B, N, num_cluster]

    # ------------------------------help function ------------------------
    def index_points(self, points, idx):
        """Sample features following the index.
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def get_neighbor_mask(self, h, w, k):
        """
        生成二维图片每个token的周围k个邻居的掩码矩阵，包括对角线的邻居。
        Args:
            h: 图片的高度
            w: 图片的宽度
            k: 邻居的半径（欧几里得距离，包含对角线）

        Returns:
            neighbor_mask: Tensor [N, N], 每个位置表示是否是邻居
        """
        N = h * w
        coords = torch.arange(N).view(h, w)  # 生成二维坐标索引
        # coords_flat = coords.flatten()  # 展平为一维

        # 获取二维坐标
        x = torch.arange(h).view(-1, 1).repeat(1, w).flatten()
        y = torch.arange(w).repeat(h).flatten()

        # 计算所有点之间的欧几里得距离
        euclidean_dist = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2 + (y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sqrt()

        # 生成邻居掩码矩阵
        neighbor_mask = euclidean_dist <= k
        return neighbor_mask


    # ------------------------------ DPC-KNN ------------------
    def cluster_dpc_knn_MCTA(self, token_dict, cluster_num, k=5, token_mask=None, H=0, W=0):
        """
        Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all tokens are regarded as meaningful.
        """
        # print(if_cosine_similarity)
        # print(if_attention_score)
        # print(attn_proj_layer)
        # print(if_distance_score)
        # print(if_feature_distance_scroe)
        # print(if_auto_weight)
        # print(weight)
        # sys.exit()

        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape       # x: [B,N,C]

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)    # dist_matrix: [N,N], (i,j)代表第i个到第j的距离是多少，这里默认是欧拉距离

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                                (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            # 第一个张量是包含最小(largest=false)的k个值。第二个张量是包含这些值在输入张量中的索引。在dim=-1上也就是C上进行的
            # 这里使用的k，也就是先选出k个最大的值，largest=False代表选出最小的几个，dim=-1是在第二个维度,防止找重复了
            # 后面使用的是dist_nearest
            # 中心的k点应该是大家的样子都比较相似的，所以在C维度上的距离最小就为中心点
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            # calculate S(i,j) based on spatial excitation
            if self.if_local_density:
                neighbor_mask = self.get_neighbor_mask(H, W, k=self.k_local).to(dist_matrix.device)  # k=1.5表示包括对角线的邻居
                neighbor_mask = neighbor_mask.unsqueeze(0).expand(B, -1, -1)  # 扩展为 [B, N, N]
                filtered_dist_matrix = dist_matrix.masked_fill(~neighbor_mask, float('inf'))
                dist_nearest_1, _ = torch.topk(-filtered_dist_matrix, k=k, dim=-1, largest=True, sorted=True)
                dist_nearest_1 = -dist_nearest_1  # 转回正值        
                if self.if_spatial_excitation and self.if_ex_density:   # using spatial excitation
                    s = self.calculate_spatial_excitation(dist_matrix, k, self.alpha)
                    density = (-( (dist_nearest_1 ** 2) * s).mean(dim=-1)).exp()
                else:   # no spatial excitation
                    density = (-(dist_nearest_1 ** 2).mean(dim=-1)).exp()
            else:   # no local density
                if self.if_spatial_excitation and self.if_ex_density:
                    s = self.calculate_spatial_excitation(dist_matrix, index_nearest, k, self.alpha)
                    density = (-( (dist_nearest ** 2) * s).mean(dim=-1)).exp()
                else:   # 对应论文not token are equal的公式1
                    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6    # [Bs, N]

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator 论文中的算distance的那个第二个公式
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            if self.if_spatial_excitation and self.if_ex_distance:
                dist = dist * self.calculate_spatial_excitation(dist_matrix, N, self.alpha)
                
            # select clustering center according to score   相乘得到最大的几个点
            score = dist * density

            # index_down是前cluster num个，指标最大的token的id，[B, cluster_num],现在的index_down代表是谁
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)    # torch.Size([10, 128])

            # original one, assign tokens to the nearest center, 
            if False:
                dist_matrix = index_points(dist_matrix, index_down)
                print(f'this is dist_matrix.size() {dist_matrix.size()}')       # torch.Size([10, 128, 7680])
                idx_cluster = dist_matrix.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
                print(f'idx_cluster.size() is {idx_cluster.size()}')        # 最好的目标就是得到这个idx_cluster.size() is torch.Size([10, 7680])

            # multiple-criterion token assignemnt (MCTA)---------------------------------------------------------
            cosine_score = 0
            attention_score = 0
            distance_score = 0
            feature_distance_score = 0
            if self.if_MCTA:
                if self.if_cosine_similarity:
                    cosine_score = self.calculate_cosine_similarity(x, index_down)  # 应该为torch.Size([10, 128, 7680])， 现在就是torch.Size([10, 128, 7680])
                if self.if_distance_score:
                    distance_score = self.calculate_distance_score(x, index_down, H, W)
                if self.if_feature_distance_scroe:
                    feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
                if self.if_attention_score:
                    attention_score = self.calculate_attention_score(x, index_down)
            else:
                feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)

        if self.if_auto_weight:
            final_score = self.cos_weight*cosine_score + self.att_weight*attention_score + \
            self.spa_dis_weight*distance_score + self.fea_dis_weight*feature_distance_score
        

        with torch.no_grad():
            # print(f'cosine score is {cosine_score}')
            # print(f'attentions_score is {attention_score}')
            # print(f'distance_score is {distance_score}')
            # torch.Size([10, 128, 7680])
            if not self.if_auto_weight:
                final_score = self.weight[0]*cosine_score + self.weight[1]*attention_score + self.weight[2]*distance_score + self.weight[3]*feature_distance_score
                # print(f'final score is {final_score.shape}')
            # ----------------------------------------------------------------------------------------------------
            # idx_cluster = final_score.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # final_score = self.index_points(final_score, index_down)       
            idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # print(f'idx_cluster.shape is {idx_cluster.shape}')
            # make sure cluster center merge to itself  [0,1...9]->[[0],[1]...[9]]->[[0，0 cluster_num个]，[1..],...]
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]
            # print(f'{idx_cluster.size()}---------------------')

        return idx_cluster, cluster_num

    def cluster_dpc_knn_MCTA_backup(self, token_dict, cluster_num, k=5, token_mask=None, H=0, W=0):
        """
        Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all tokens are regarded as meaningful.
        """
        # print(if_cosine_similarity)
        # print(if_attention_score)
        # print(attn_proj_layer)
        # print(if_distance_score)
        # print(if_feature_distance_scroe)
        # print(if_auto_weight)
        # print(weight)
        # sys.exit()

        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape       # x: [B,N,C]

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)    # dist_matrix: [N,N], (i,j)代表第i个到第j的距离是多少，这里默认是欧拉距离

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                                (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            # 第一个张量是包含最小(largest=false)的k个值。第二个张量是包含这些值在输入张量中的索引。在dim=-1上也就是C上进行的
            # 这里使用的k，也就是先选出k个最大的值，largest=False代表选出最小的几个，dim=-1是在第二个维度,防止找重复了
            # 后面使用的是dist_nearest
            # 中心的k点应该是大家的样子都比较相似的，所以在C维度上的距离最小就为中心点
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            # calculate S(i,j) based on spatial excitation
            if self.if_spatial_excitation and self.if_ex_density:
                s = self.calculate_spatial_excitation(dist_matrix, index_nearest, k, self.alpha)
                density = (-( (dist_nearest ** 2) * s).mean(dim=-1)).exp()
            else:
                # 对应论文not token are equal的公式1
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6    # [Bs, N]

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator 论文中的算distance的那个第二个公式
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            if self.if_spatial_excitation and self.if_ex_distance:
                dist = dist * self.calculate_spatial_excitation(dist_matrix, index_nearest, N, self.alpha)
                
            # select clustering center according to score   相乘得到最大的几个点
            score = dist * density

            # index_down是前cluster num个，指标最大的token的id，[B, cluster_num],现在的index_down代表是谁
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)    # torch.Size([10, 128])

            # original one, assign tokens to the nearest center, 
            if False:
                dist_matrix = index_points(dist_matrix, index_down)
                print(f'this is dist_matrix.size() {dist_matrix.size()}')       # torch.Size([10, 128, 7680])
                idx_cluster = dist_matrix.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
                print(f'idx_cluster.size() is {idx_cluster.size()}')        # 最好的目标就是得到这个idx_cluster.size() is torch.Size([10, 7680])

            # multiple-criterion token assignemnt (MCTA)---------------------------------------------------------
            cosine_score = 0
            attention_score = 0
            distance_score = 0
            feature_distance_score = 0
            if self.if_MCTA:
                if self.if_cosine_similarity:
                    cosine_score = self.calculate_cosine_similarity(x, index_down)  # 应该为torch.Size([10, 128, 7680])， 现在就是torch.Size([10, 128, 7680])
                if self.if_distance_score:
                    distance_score = self.calculate_distance_score(x, index_down, H, W)
                if self.if_feature_distance_scroe:
                    feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
                if self.if_attention_score:
                    attention_score = self.calculate_attention_score(x, index_down)
            else:
                feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)

        if self.if_auto_weight:
            final_score = self.cos_weight*cosine_score + self.att_weight*attention_score + \
            self.spa_dis_weight*distance_score + self.fea_dis_weight*feature_distance_score
        

        with torch.no_grad():
            # print(f'cosine score is {cosine_score}')
            # print(f'attentions_score is {attention_score}')
            # print(f'distance_score is {distance_score}')
            # torch.Size([10, 128, 7680])
            if not self.if_auto_weight:
                final_score = self.weight[0]*cosine_score + self.weight[1]*attention_score + self.weight[2]*distance_score + self.weight[3]*feature_distance_score
                # print(f'final score is {final_score.shape}')
            # ----------------------------------------------------------------------------------------------------
            # idx_cluster = final_score.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # print(f'idx_cluster.shape is {idx_cluster.shape}')
            # make sure cluster center merge to itself  [0,1...9]->[[0],[1]...[9]]->[[0，0 cluster_num个]，[1..],...]
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]
            # print(f'{idx_cluster.size()}---------------------')

        return idx_cluster, cluster_num

    def cluster_dpc_knn(self, token_dict, cluster_num, k=5, token_mask=None):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
                total time 0.12060689926147461  local density 0.0010237693786621094  t2 0.1197960376739502
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)        # 0.119
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    # ----------------- SNN-DPC ------------------------------------------------------------
    def snn_dpc_backup(self, token_dict, cluster_num):
            """
            Perform SNN-DPC clustering using formulas from the paper.

            Args:
                data (torch.Tensor): Input tensor of shape (B, N, C).

            Returns:
                torch.Tensor: Cluster assignments of shape (B, N).
            """
            with torch.no_grad():
                x = token_dict['x']
                B, N, C = x.shape

                # Step 1: Normalize data 对应原文algorithm 1.1 normalization，但是这步不重要，所以注释掉了
                # data = (data - data.mean(dim=2, keepdim=True)) / data.std(dim=2, keepdim=True)

                # Step 2: Compute pairwise distances (B, N, N) 对应原文algorithm 1.2 计算distance matrix
                dist_matrix = torch.cdist(x, x)

                # Step 3: Compute shared nearest neighbors (SNN) similarity 对应原文algorithm 1.3 calculate similarity matrix, by Eq. 7
                # Find K-nearest neighbors
                _, knn_indices = torch.topk(-dist_matrix, k=self.k, dim=2)      # 取负是为了找最小的，也就是最近的topk个
                knn_mask = torch.zeros_like(dist_matrix).scatter_(2, knn_indices, 1)  # Binary mask for KNN
                shared_neighbors = knn_mask @ knn_mask.transpose(1, 2)

                # 注意这里的snn_similarity的分母时两个点到各自KNN的点的距离之和，不是Eq7中的SNN的点距离之和 ！！！！！！！！！！！！！！！！！！！！！！！！！！！
                # i_j_knn_sum = (dist_matrix.gather(2, knn_indices).sum(dim=2)[:, :, None] + dist_matrix.gather(2, knn_indices).sum(dim=2)[:, None, :])
                knn_dist = dist_matrix.gather(2, knn_indices)  # (B, N, k)
                i_j_knn_sum = (knn_dist.sum(dim=2)[:, :, None] + knn_dist.sum(dim=2)[:, None, :])
                snn_similarity = (shared_neighbors**2) / (i_j_knn_sum + 1e-8)

                # Step 4: Calculate local density ρ (Eq. 9)
                local_density = snn_similarity.sum(dim=2)
                local_density.add_(torch.rand(local_density.shape, device=local_density.device, dtype=local_density.dtype) * 1e-8)
                # print(f'local_density: {local_density}')

                # Step 5: Calculate distance δ to the nearest larger density point (Eq. 10)
                density_diff = local_density[:, :, None] - local_density[:, None, :]
                # get mask, Eq.9 only need ρj>ρi
                higher_density_mask = (density_diff < 0).float()
                # get the mulplication value of Eq.9
                delta = dist_matrix * i_j_knn_sum * higher_density_mask
                delta[delta == 0] = 1000  # replace the value of zero, and simplify Eq.11(just give it a very big value) 这里与Eq11不一样
                delta, _ = delta.min(dim=2) # get the min in Eq.9

                # Step 6: Compute decision value γ = ρ × δ
                score = local_density * delta

                # just follow DPC-KNN's implementation
                _, index_down = torch.topk(score, k=cluster_num, dim=-1)

                # assign tokens to the nearest center
                dist_matrix = self.index_points(dist_matrix, index_down)

                idx_cluster = dist_matrix.argmin(dim=1)

                # make sure cluster center merge to itself
                idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
                idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
                idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

            return idx_cluster, cluster_num

    def snn_dpc(self, token_dict, cluster_num):
            """
            Perform SNN-DPC clustering using formulas from the paper.

            Args:
                data (torch.Tensor): Input tensor of shape (B, N, C).

            Returns:
                torch.Tensor: Cluster assignments of shape (B, N).
            """
            with torch.no_grad():
                x = token_dict['x']
                B, N, C = x.shape

                # Step 2: Compute pairwise distances (B, N, N)
                dist_matrix = torch.cdist(x, x)

                # Step 3: Compute shared nearest neighbors (SNN) similarity
                # Find K-nearest neighbors (in-place to save memory)
                _, knn_indices = torch.topk(-dist_matrix, k=self.k, dim=2)
                knn_mask = torch.zeros_like(dist_matrix).scatter_(2, knn_indices, 1)  # in-place scatter operation
                
                # Calculate shared neighbors in-place to save memory
                shared_neighbors = knn_mask @ knn_mask.transpose(1, 2)

                # Step 3 (continued): Compute the denominator for SNN similarity
                knn_dist = dist_matrix.gather(2, knn_indices)  # (B, N, k)
                i_j_knn_sum = (knn_dist.sum(dim=2)[:, :, None] + knn_dist.sum(dim=2)[:, None, :])
                
                # Compute SNN similarity in-place to save memory
                snn_similarity = shared_neighbors**2 / (i_j_knn_sum + 1e-8)

                # Step 4: Calculate local density ρ (Eq. 9)
                # Sum over neighbors
                local_density = snn_similarity.sum(dim=2)

                # To avoid potential NaN issues, add a small noise in-place
                local_density.add_(torch.rand(local_density.shape, device=local_density.device, dtype=local_density.dtype) * 1e-8)

                # Step 5: Calculate distance δ to the nearest larger density point (Eq. 10)
                density_diff = local_density[:, :, None] - local_density[:, None, :]
                higher_density_mask = (density_diff < 0).float()

                # In-place multiplication to calculate delta
                delta = dist_matrix * i_j_knn_sum * higher_density_mask

                # Replace zero values with a large value (to handle cases where delta might be zero)
                delta[delta == 0] = 1000

                # Compute the minimum value along the third dimension to get δ
                delta, _ = delta.min(dim=2)

                # Step 6: Compute decision value γ = ρ × δ
                score = local_density * delta

                # Perform clustering by selecting top-k scores (efficient)
                _, index_down = torch.topk(score, k=cluster_num, dim=-1)

                # Assign tokens to the nearest center
                dist_matrix = self.index_points(dist_matrix, index_down)

                # Find the cluster assignments (argmin over distance matrix)
                idx_cluster = dist_matrix.argmin(dim=1)

                # Ensure the cluster centers merge to themselves
                idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
                idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
                idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

            return idx_cluster, cluster_num

     # ----------------- SNN-DPC ------------------------------------------------------------
    
    # ----------------- stardard varation weighted ------------------------------------------------------------
    def cluster_dpc_knn_StdVar(self, token_dict, cluster_num, k=5, token_mask=None):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            # std variation weighted x
            # Step 1: Compute mean and standard deviation for each feature across tokens
            mean_features = x.mean(dim=1, keepdim=True)  # Shape: [B, 1, C]
            std_features = torch.sqrt(((x - mean_features) ** 2).mean(dim=1, keepdim=True))  # Shape: [B, 1, C]
            # Step 2: Compute feature weights
            weights = std_features / std_features.sum(dim=-1, keepdim=True)  # Shape: [B, 1, C]
            # Step 3: Compute weighted distance matrix
            x = x * weights  # Apply weights to features
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num

    # ----------------- tcformerv2 -------------------------------------
    def cluster_dpc_knnv2(self, token_dict, cluster_num, k=5, token_mask=None):
        """Cluster tokens with DPC-KNN algorithm.
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate the whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
                total time 0.12060689926147461  local density 0.0010237693786621094  t2 0.1197960376739502
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            
            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)        # 0.119
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    # ----------------- fuzzy knn ---------------------------------------
    def cluster_dpc_fknn_backup(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            t1 = time.time()
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)        # t1: 0.0006787776947021484 
            print(f't1: {time.time() - t1} ')

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            t2 = time.time()
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            print(f't2: {time.time() - t2} ')
            t3 = time.time()
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev
            print(f't3: {time.time() - t3} ')
            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            t4 = time.time()
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            print(f't4: {time.time() - t4} ')
            t5 = time.time()
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            print(f't5: {time.time() - t5} ')
            t6 = time.time()
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            print(f't6: {time.time() - t6} ')
            t7 = time.time()
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            print(f't7: {time.time() - t7} ')
            t8 = time.time()
            nn_mask[batch_indices, token_indices, nn_indices] = True
            print(f't8: {time.time() - t8} ')
            # u
            t9 = time.time()
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            print(f't9: {time.time() - t9} ')
            t10 = time.time()
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            print(f't10: {time.time() - t10} ')
            t11 = time.time()
            mu = torch.where(nn_mask, mu, mu_non_nn)
            print(f't11: {time.time() - t11} ')

            # Step 4: Calculate local density ρ (Definition 2)
            t12 = time.time()
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            print(f't12: {time.time() - t12} ')
            t13 = time.time()
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            print(f't13: {time.time() - t13} ')
            t14 = time.time()
            rho = rho_nn + rho_non_nn
            print(f't14: {time.time() - t14} ')

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens
            print(f' local density is {time.time() - t1}')
            # Step 5: Compute distance indicator for clustering
            t15 = time.time()
            mask = rho[:, None, :] > rho[:, :, None]
            print(f't15: {time.time() - t15} ')
            t16 = time.time()
            mask = mask.type(x.dtype)
            print(f't16: {time.time() - t16} ')
            t17 = time.time()
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            print(f't17: {time.time() - t17} ')
            t18 = time.time()
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
            print(f't18: {time.time() - t18} ')

            # Step 6: Select clustering centers
            t19 = time.time()
            score = dist * rho
            print(f't19: {time.time() - t19} ')
            t20 = time.time()
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            print(f't20: {time.time() - t20} ')

            # Step 7: Assign tokens to nearest cluster centers
            t21 = time.time()
            dist_matrix = self.index_points(dist_matrix, index_down)
            print(f't21: {time.time() - t21} ')
            t22 = time.time()
            idx_cluster = dist_matrix.argmin(dim=1)
            print(f't22: {time.time() - t22} ')

            # Ensure cluster centers map to themselves
            t23 = time.time()
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            print(f't23: {time.time() - t23} ')
            t24 = time.time()
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            print(f't24: {time.time() - t24} ')
            t25 = time.time()
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
            print(f't25: {time.time() - t25} ')
        print(f'total {time.time()-t1}')
        sys.exit()
        return idx_cluster, cluster_num

    def cluster_dpc_fknn(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).
        total: 0.25165724754333496
        local density is 0.001909494400024414
        t21: 0.24913883209228516
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Assign tokens to nearest cluster centers
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    def cluster_dpc_fknn_MCTA(self, token_dict, cluster_num, k=5, token_mask=None, H=0, W=0):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included). + MCTA
        total time is 0.28340601921081543
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            cosine_score = 0
            attention_score = 0
            distance_score = 0
            feature_distance_score = 0
            if self.if_MCTA:
                if self.if_cosine_similarity:
                    cosine_score = self.calculate_cosine_similarity(x, index_down)  # 应该为torch.Size([10, 128, 7680])， 现在就是torch.Size([10, 128, 7680])
                if self.if_distance_score:
                    distance_score = self.calculate_distance_score(x, index_down, H, W)
                if self.if_feature_distance_scroe:
                    feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
                if self.if_attention_score:
                    attention_score = self.calculate_attention_score(x, index_down)
            else:
                feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)

        if self.if_auto_weight:
            final_score = self.cos_weight*cosine_score + self.att_weight*attention_score + \
            self.spa_dis_weight*distance_score + self.fea_dis_weight*feature_distance_score
        

        with torch.no_grad():
            # print(f'cosine score is {cosine_score}')
            # print(f'attentions_score is {attention_score}')
            # print(f'distance_score is {distance_score}')
            # torch.Size([10, 128, 7680])
            if not self.if_auto_weight:
                final_score = self.weight[0]*cosine_score + self.weight[1]*attention_score + self.weight[2]*distance_score + self.weight[3]*feature_distance_score
                # print(f'final score is {final_score.shape}')
            # ----------------------------------------------------------------------------------------------------
            # idx_cluster = final_score.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # print(f'idx_cluster.shape is {idx_cluster.shape}')
            # make sure cluster center merge to itself  [0,1...9]->[[0],[1]...[9]]->[[0，0 cluster_num个]，[1..],...]
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]
            # print(f'{idx_cluster.size()}---------------------')
        return idx_cluster, cluster_num
    
    def cluster_dpc_fknn_StdVar(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).
        + stadard varation weighted
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # std var weighted
            mean_features = x.mean(dim=1, keepdim=True)  # Shape: [B, 1, C]
            std_features = torch.sqrt(((x - mean_features) ** 2).mean(dim=1, keepdim=True))  # Shape: [B, 1, C]
            # Step 2: Compute feature weights
            weights = std_features / std_features.sum(dim=-1, keepdim=True)  # Shape: [B, 1, C]
            # Step 3: Compute weighted distance matrix
            x = x * weights  # Apply weights to features

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Assign tokens to nearest cluster centers
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num
   
    def cluster_dpc_fknn_excitation(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).
        total: 0.25165724754333496
        local density is 0.001909494400024414
        t21: 0.24913883209228516
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            if self.if_ex_density and self.if_spatial_excitation:
                mu = mu * self.calculate_spatial_excitation(dist_matrix, N, self.alpha)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            if self.if_spatial_excitation and self.if_ex_distance:
                dist = dist * self.calculate_spatial_excitation(dist_matrix, N, self.alpha)
            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Assign tokens to nearest cluster centers
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        return idx_cluster, cluster_num

    def cluster_dpc_fknn_wsn(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Calculate W and WSN
            # W = calculate_W(dist_matrix, nn_indices, k)
            d_ij_knn = torch.gather(
                dist_matrix.unsqueeze(2).expand(-1, -1, N, -1),  # (B, N, N, N)
                dim=3,
                index=nn_indices.unsqueeze(1).expand(-1, N, -1, -1)     # (B, N, N, k)
            )  # 输出形状 (B, N, N, k)

            W= (1 / (d_ij_knn + 1)).sum(dim=-1)     # (B, N, N)
            W = W + W.transpose(1,2)
            # 老方法        
            knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)  # in-place scatter operation
            # Calculate shared neighbors in-place to save memory
            snn = knn_mask @ knn_mask.transpose(1, 2)

            WSN = -snn * W * 100
            WSN_score = WSN + dist_matrix

            dist_matrix = self.index_points(WSN_score, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num

    def cluster_dpc_fknn_wsn_simplify_v1(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).并不是原来的
        simplify是指在wsn这个assignment中 我把WSN指标改成了 i的knn到i的距离 j的knn到j的距离。。而不是i的knn到j的距离 j的knn到i的距离
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            nn_mask[batch_indices, token_indices, nn_indices] = True

            # u
            mu = (torch.exp(- (dist_matrix)**2)/ (dist_matrix + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2
            mu = torch.where(nn_mask, mu, mu_non_nn)

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = (mu * nn_mask).sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            rho_non_nn = (mu * (~nn_mask)).sum(dim=-1) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Calculate W and WSN
            # W = calculate_W(dist_matrix, nn_indices, k)
            # batch_indices = torch.arange(B, device=dist_matrix.device)[:, None, None].expand(B, N, k)
            # token_indices = torch.arange(N, device=dist_matrix.device)[None, :, None].expand(B, N, k)
            # nn_distances = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)

            # # Calculate the first term: sum over KNN(i)
            # W_first_term = (1 / (nn_distances + 1)).sum(dim=-1)  # Shape: (B, N)
            batch_indices = torch.arange(B, device=dist_matrix.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=dist_matrix.device)[None, :, None].expand(B, N, k)
            nn_distances = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            W = (1 / (nn_distances + 1)).sum(dim=-1)  # Shape: (B, N)
            W = W[:, :, None] + W[:, None, :] 

            # 老方法        
            knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)  # in-place scatter operation
            # Calculate shared neighbors in-place to save memory
            snn = knn_mask @ knn_mask.transpose(1, 2)

            WSN = -snn * W * 100
            WSN_score = WSN + dist_matrix

            dist_matrix = self.index_points(WSN_score, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num

    def cluster_dpc_fknn_wsn_simplify_v2(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included).并不是原来的
        simplify是指在wsn这个assignment中 我把WSN指标改成了 i的knn到i的距离 j的knn到j的距离。。而不是i的knn到j的距离 j的knn到i的距离
        优化了内存 不过是一些小点，不重要
        v2是指还把fknn给改了 在fuzzy部分 KNN以外的部分改成了包括knn内在的 => 相当于加强了topk的比重
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            # nn_mask[batch_indices, token_indices, nn_indices] = True      # 不需要这个nn_mask可以省内存

            # u
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            mu = (torch.exp(- (dist_selected)**2)/ (dist_selected + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = mu.sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            # rho_non_nn = ((mu * (~nn_mask)).sum(dim=-1) - rho_nn) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho_non_nn = mu_non_nn.sum(-1) / N      # 不是按照原来的公式，但是可以省内存
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # Step 7: Calculate W and WSN
            # W = calculate_W(dist_matrix, nn_indices, k)
            # batch_indices = torch.arange(B, device=dist_matrix.device)[:, None, None].expand(B, N, k)
            # token_indices = torch.arange(N, device=dist_matrix.device)[None, :, None].expand(B, N, k)
            # nn_distances = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)

            # # Calculate the first term: sum over KNN(i)
            # W_first_term = (1 / (nn_distances + 1)).sum(dim=-1)  # Shape: (B, N)
            # nn_distances = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            # 为了节约内存，修改了一下这里的算法，我的W不是i的KNN到j的距离，而是i的KNN到i的距离
            # 使用函数
            # W = (1 / (dist_selected + 1)).sum(dim=-1)  # Shape: (B, N)  把单独能的，换成上面用过的dist_selected，即只有knn的距离
            # W = W.unsqueeze(-1) + W.unsqueeze(-2)  # [优化] 避免 `W[:, :, None] + W[:, None, :]` 造成大规模广播

            # # 老方法        
            # knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)  # in-place scatter operation
            # # Calculate shared neighbors in-place to save memory
            # snn = knn_mask @ knn_mask.transpose(1, 2)

            WSN = -100*self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            score = self.index_points(dist_matrix, index_down) + WSN
            idx_cluster = score.argmin(dim=1)

            # Ensure cluster centers map to themselves
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num

    def calculate_WSN(self, dist_selected, dist_matrix, nn_indices, index_down):
        W = (1 / (dist_selected + 1)).sum(dim=-1)  # Shape: (B, N)  把单独能的，换成上面用过的dist_selected，即只有knn的距离
        W = W.unsqueeze(-1) + W.unsqueeze(-2)  # [优化] 避免 `W[:, :, None] + W[:, None, :]` 造成大规模广播
    
        knn_mask = torch.zeros_like(dist_matrix).scatter_(2, nn_indices, 1)  # in-place scatter operation
        # Calculate shared neighbors in-place to save memory
        snn = knn_mask @ knn_mask.transpose(1, 2)

        WSN = snn * W
        WSN = self.index_points(WSN, index_down)
        return WSN

    def cluster_dpc_fknnv2_WSN_MCTA(self, token_dict, cluster_num, k=5, token_mask=None, H=0, W=0):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included). + MCTA
        total time is 0.28340601921081543
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            # nn_mask[batch_indices, token_indices, nn_indices] = True      # 不需要这个nn_mask可以省内存

            # u
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            mu = (torch.exp(- (dist_selected)**2)/ (dist_selected + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = mu.sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            # rho_non_nn = ((mu * (~nn_mask)).sum(dim=-1) - rho_nn) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho_non_nn = mu_non_nn.sum(-1) / N      # 不是按照原来的公式，但是可以省内存
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)


            cosine_score = 0
            attention_score = 0
            distance_score = 0
            feature_distance_score = 0
            WSN_score = 0
            if self.if_MCTA:
                if self.if_cosine_similarity:
                    cosine_score = self.calculate_cosine_similarity(x, index_down)  # 应该为torch.Size([10, 128, 7680])， 现在就是torch.Size([10, 128, 7680])
                if self.if_distance_score:
                    distance_score = self.calculate_distance_score(x, index_down, H, W)
                if self.if_feature_distance_scroe:
                    feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
                if self.if_attention_score:
                    attention_score = self.calculate_attention_score(x, index_down)
                WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            else:
                feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)


            final_score = self.weight[0]*cosine_score + self.weight[1]*attention_score + self.weight[2]*distance_score + self.weight[3]*feature_distance_score + self.weight[4]*WSN_score
            # ----------------------------------------------------------------------------------------------------
            # idx_cluster = final_score.argmin(dim=1)     # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # print(f'idx_cluster.shape is {idx_cluster.shape}')
            # make sure cluster center merge to itself  [0,1...9]->[[0],[1]...[9]]->[[0，0 cluster_num个]，[1..],...]
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]
            # print(f'{idx_cluster.size()}---------------------')
        return idx_cluster, cluster_num
      
    def cluster_dpc_fknnv2_WSN_MCTA_seq(self, token_dict, cluster_num, k=5, token_mask=None, H=0, W=0):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included). + MCTA
        total time is 0.28340601921081543
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            # nn_mask[batch_indices, token_indices, nn_indices] = True      # 不需要这个nn_mask可以省内存

            # u
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            mu = (torch.exp(- (dist_selected)**2)/ (dist_selected + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = mu.sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            # rho_non_nn = ((mu * (~nn_mask)).sum(dim=-1) - rho_nn) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho_non_nn = mu_non_nn.sum(-1) / N      # 不是按照原来的公式，但是可以省内存
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            cosine_score = 0
            attention_score = 0
            distance_score = 0
            feature_distance_score = 0
            KL_score=0
            SSIM_score=0
            WSN_score = 0
            if self.if_MCTA:
                if self.if_cosine_similarity:
                    cosine_score = self.calculate_cosine_similarity(x, index_down)  # 应该为torch.Size([10, 128, 7680])， 现在就是torch.Size([10, 128, 7680])
                if self.if_distance_score:
                    distance_score = self.calculate_distance_score(x, index_down, H, W)
                if self.if_feature_distance_scroe:
                    feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)
                if self.if_attention_score:
                    attention_score = self.calculate_attention_score(x, index_down)
                if self.if_KL_score:
                    KL_score = self.calculate_KL_score(x, index_down)
                if self.if_SSIM_score:
                    SSIM_score = self.calculate_SSIM_score(x, index_down)
                WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)

            else:
                feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)

            final_score = self.weight[0]*cosine_score + self.weight[1]*attention_score + self.weight[2]*distance_score + self.weight[3]*feature_distance_score + 100000*WSN_score + self.weight[4]*KL_score + self.weight[5]*SSIM_score

            idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            # _, indices = final_score.topk(k=2, dim=1)  # [B, 2, N]
            # print(f'idx_cluster.shape is {idx_cluster.shape}')
            # make sure cluster center merge to itself  [0,1...9]->[[0],[1]...[9]]->[[0，0 cluster_num个]，[1..],...]
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)    # [0,1]->[[0,1,...], [0,1,2...],....]
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]
            # print(f'{idx_cluster.size()}---------------------')
        return idx_cluster, cluster_num

  
    def cluster_dpc_fknnv2_WSNseq_FMerge(self, token_dict, cluster_num, k=5, token_mask=None):
        """
        Cluster tokens with DPC-FWSN algorithm (Definition 1 and 2 included). + MCTA
        total time is 0.28340601921081543
        """
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape
            # Step 1: Compute pairwise distance
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # Step 2: Calculate sparsity factor (phi) and mean distance (omega)
            omega = dist_matrix.mean(dim=(1, 2), keepdim=True)  # Global mean distance
            phi = torch.sqrt(((dist_matrix - omega) ** 2).mean(dim=(1, 2), keepdim=True))  # Std dev

            # Step 3: Calculate membership function μ(i, j) (Definition 1)
            _, nn_indices = torch.topk(-dist_matrix, k=k, dim=-1)  # Nearest neighbors
            nn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            batch_indices = torch.arange(B, device=x.device)[:, None, None].expand(B, N, k)
            token_indices = torch.arange(N, device=x.device)[None, :, None].expand(B, N, k)
            # nn_mask[batch_indices, token_indices, nn_indices] = True      # 不需要这个nn_mask可以省内存

            # u
            dist_selected = dist_matrix[batch_indices, token_indices, nn_indices]  # Shape: (B, N, k)
            mu = (torch.exp(- (dist_selected)**2)/ (dist_selected + 1))**2
            mu_non_nn = torch.exp(-(phi * dist_matrix)**2)/ (dist_matrix + 1)**2

            # Step 4: Calculate local density ρ (Definition 2)
            rho_nn = mu.sum(dim=-1) / k  # 最近邻的模糊隶属度贡献
            # rho_non_nn = ((mu * (~nn_mask)).sum(dim=-1) - rho_nn) / (N - k)  # 非最近邻的模糊隶属度贡献
            rho_non_nn = mu_non_nn.sum(-1) / N      # 不是按照原来的公式，但是可以省内存
            rho = rho_nn + rho_non_nn

            if token_mask is not None:
                rho = rho * token_mask  # Mask out invalid tokens

            # Step 5: Compute distance indicator for clustering
            mask = rho[:, None, :] > rho[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # Step 6: Select clustering centers
            score = dist * rho
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)


            WSN_score = self.calculate_WSN(dist_selected, dist_matrix, nn_indices, index_down)
            feature_distance_score = self.calculate_feature_distance_score(dist_matrix, index_down)

            final_score = feature_distance_score + 100000*WSN_score 

            # idx_cluster = final_score.argmax(dim=1)       # 输出的是每列(第1维)的最小值的索引 [B,N]， torch.Size([10, 128, 7680])
            _, idx_cluster = final_score.topk(k=self.k_merge, dim=1)  # [B, 2, N]   改了这里
            # idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)     
            # idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)    # [0,1]->[[0,1,...], [0,1,2...],....]
            # idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)        # 应该为tensor:[10, 7680]

        return idx_cluster, cluster_num
  

class CTM_backup(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        token_dict = token_dict.copy()
        x = self.conv(token_dict)
        x = self.norm(x)
        token_score = self.score(x)
        token_weight = token_score.exp()

        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        cluster_num = max(math.ceil(N * self.sample_ratio), 1)
        idx_cluster, cluster_num = self.cluster_dpc_knnn(
            token_dict, cluster_num, self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        H, W = token_dict['map_size']
        H = math.floor((H - 1) / 2 + 1)
        W = math.floor((W - 1) / 2 + 1)
        down_dict['map_size'] = [H, W]

        return down_dict, token_dict

    def cluster_dpc_knnn(self, token_dict, cluster_num, k=5, token_mask=None):
        with torch.no_grad():
            x = token_dict['x']
            B, N, C = x.shape

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)

            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num
    
    def index_points(self, points, idx):
        """Sample features following the index.
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


