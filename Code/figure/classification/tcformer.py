import sys
import os
sys.path.insert(0, '..')
import torch.nn as nn
from timm.models.registry import register_model
from tcformer_module.tcformer import TCFormer as _TCFormer
from functools import partial
import sys
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np

class TCFormer(_TCFormer):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        # classification head
        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(self._init_weights)

        self.i = 0

    def forward(self, x):
        # x_copy = x  # for visualization
        x = self.forward_features(x)
        self.i = self.i+1
        # # for visualization
        # self.save_tensor_as_images(x_copy, save_dir='/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/', n=0)
        # vis_img = self.vis_tokens(x_copy, token_dict=x[0])
        # self.save_tensor_as_images(vis_img, save_dir='/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/', n=1)
        # vis_img = self.vis_tokens(x_copy, token_dict=x[1])
        # self.save_tensor_as_images(vis_img, save_dir='/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/', n=2)
        # vis_img = self.vis_tokens(x_copy, token_dict=x[2])
        # self.save_tensor_as_images(vis_img, save_dir='/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/', n=3)
        # vis_img = self.vis_tokens(x_copy, token_dict=x[3])
        # self.save_tensor_as_images(vis_img, save_dir='/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/', n=4)
        # self.combine_images(
        #     load_path = '/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/ori/ori'+str(self.i)+'/',
        #     save_path = '/home/jitri7/muyi/TCFormer-master/classification/cluster_vis_image/TCFormer/combined/', 
        #     output_name=str(self.i)+'.jpg')
        # if self.i == 300:
        #     sys.exit()


        x = x[-1]['x'].mean(dim=1)
        x = self.head(x)
        return x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def vis_tokens(self, img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
        """Visualize tokens
        Return:
            vis_img (Tensor[B, 3, H, W]): visualize result.

        Args:
            img (Tensor[B, 3, H, W]): input image.
            token_dict (dict): dict for input token information
            edge_color (float[int]): color for edges
            edge_width (int): width for edges
        """

        N = token_dict['token_num']
        device, dtype = img.device, img.dtype

        # color_map = torch.tensor(img, device=device, dtype=float) / 255.0
        # color_map = color_map.permute(2, 0, 1)[None, ...]
        color_map = F.avg_pool2d(img, kernel_size=4)
        B, C, H, W = color_map.shape

        token_color = self.map2token(color_map, token_dict)
        tmp_dict = token_dict.copy()
        tmp_dict['map_size'] = [H, W]
        tmp_dict['x'] = token_color
        vis_img = self.token2map(tmp_dict)

        token_idx = torch.arange(N, device=device)[None, :, None].float() / N
        tmp_dict['x'] = token_idx
        idx_map = self.token2map(tmp_dict)  # [B, 1, H, W]

        vis_img = F.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
        idx_map = F.interpolate(idx_map, [H * 8, W * 8], mode='nearest')

        kernel = idx_map.new_zeros([4, 1, 3, 3])
        kernel[:, :, 1, 1] = 1
        kernel[0, :, 0, 1] = -1
        kernel[1, :, 2, 1] = -1
        kernel[2, :, 1, 0] = -1
        kernel[3, :, 1, 2] = -1

        for i in range(edge_width):
            edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
            edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
            idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map

        edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
        vis_img = vis_img * (~edge_map) + edge_color * edge_map
        return vis_img

    def token2map(self, token_dict):
        """Transform vision tokens to feature map. This function only
        works when the resolution of the feature map is not higher than
        the initial grid structure.
        Returns:
            x_out (Tensor[B, C, H, W]): feature map.

        Args:
            token_dict (dict): dict for token information.
        """
        x = token_dict['x']
        H, W = token_dict['map_size']
        H_init, W_init = token_dict['init_grid_size']
        idx_token = token_dict['idx_token']
        B, N, C = x.shape
        N_init = H_init * W_init
        device = x.device

        if N_init == N and N == H * W:
            # for the initial tokens with grid structure, just reshape
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # for each initial grid, get the corresponding index in
        # the flattened feature map.
        idx_hw = self.get_grid_index(
            [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)  # [B, N_init]
        idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)  # [B, N_init]
        value = x.new_ones(B * N_init)

        # choose the way with fewer flops.
        if N_init < N * H * W:
            # use sparse matrix multiplication
            # Flops: B * N_init * (C+2)
            idx_hw = idx_hw + idx_batch * H * W  # [B, N_init]
            idx_tokens = idx_token + idx_batch * N  # [B, N_init]
            coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)  # [2, B * N_init]

            # torch.sparse do not support fp16
            with torch.cuda.amp.autocast(enabled=False):
                # torch.sparse do not support gradient for
                # sparse tensor, so we detach it
                value = value.detach().float()

                # build a sparse matrix with the shape [B * H * W, B * N]
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # normalize the weight for each row
                all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
                value = value / all_weight[idx_hw.reshape(-1), 0]

                # update the matrix with normalize weight
                A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

                # sparse matrix multiplication
                x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

        else:
            # use dense matrix multiplication
            # Flops: B * N * H * W * (C+2)
            coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

            # build a matrix with shape [B, H*W, N]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
            # normalize the weight
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

            x_out = A @ x  # [B, H*W, C]

        x_out = x_out.type(x.dtype)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

    def map2token(self, feature_map, token_dict):
        """Transform feature map to vision tokens. This function only
        works when the resolution of the feature map is not higher than
        the initial grid structure.

        Returns:
            out (Tensor[B, N, C]): token features.

        Args:
            feature_map (Tensor[B, C, H, W]): feature map.
            token_dict (dict): dict for token information.
        """
        idx_token = token_dict['idx_token']
        N = token_dict['token_num']
        H_init, W_init = token_dict['init_grid_size']
        N_init = H_init * W_init

        # agg_weight = token_dict['agg_weight'] if 'agg_weight' in token_dict.keys() else None
        agg_weight = None  # we do not use the weight value here

        B, C, H, W = feature_map.shape
        device = feature_map.device

        if N_init == N and N == H * W:
            # for the initial tokens with grid structure, just reshape
            return feature_map.flatten(2).permute(0, 2, 1).contiguous()

        idx_hw = self.get_grid_index(
            [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)

        idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
        if agg_weight is None:
            value = feature_map.new_ones(B * N_init)
        else:
            value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

        # choose the way with fewer flops.
        if N_init < N * H * W:
            # use sparse matrix multiplication
            # Flops: B * N_init * (C+2)
            idx_token = idx_token + idx_batch * N
            idx_hw = idx_hw + idx_batch * H * W
            indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

            # torch.sparse do not support fp16
            with torch.cuda.amp.autocast(enabled=False):
                # sparse mm do not support gradient for sparse matrix
                value = value.detach().float()
                # build a sparse matrix with shape [B*N, B*H*W]
                A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
                # normalize the matrix
                all_weight = A @ torch.ones(
                    [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
                value = value / all_weight[idx_token.reshape(-1), 0]

                A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
                # out: [B*N, C]
                out = A @ feature_map. \
                    permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
        else:
            # use dense matrix multiplication
            # Flops: B * N * H * W * (C+2)
            indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
            value = value.detach()  # To reduce the training time, we detach here.
            A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
            # normalize the matrix
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

            out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        out = out.type(feature_map.dtype)
        out = out.reshape(B, N, C)
        return out
    
    def get_grid_index(self, init_size, map_size, device):
        """For each initial grid, get its index in the feature map.
        Returns:
            idx (LongTensor[B, N_init]): index in flattened feature map.

        Args:
            init_grid_size(list[int] or tuple[int]): initial grid resolution in
                format [H_init, W_init].
            map_size(list[int] or tuple[int]): feature map resolution in format
                [H, W].
            device: the device of output
        """
        H_init, W_init = init_size
        H, W = map_size
        idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
        idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
        return idx.flatten()

    def save_tensor_as_images(self, tensor, save_dir, n=1):
        """
        将大小为 (B, 3, H, W) 的 Tensor 拆开并保存为图片。

        Args:
            tensor (torch.Tensor): 输入的 Tensor 大小为 (B, 3, H, W)。
            save_dir (str): 保存图片的目录。
        """
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # B = tensor.size(0)
        # for i in range(B):
        #     img_tensor = tensor[i]  # 大小为 (3, H, W) # 获取第 i 张图片
        #     img = to_pil_image(img_tensor)  # 将 Tensor 转换为 PIL 图像
        #     img.save(os.path.join(save_dir, f'image_{i}.png'))  # 保存图片
        img = tensor[0]
        img = to_pil_image(img)  # 将 Tensor 转换为 PIL 图像
        img.save(os.path.join(save_dir, f'image_{n}.png'))  # 保存图片

    def combine_images(self, load_path, save_path, gap=10, output_name="combined.jpg"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 读取目录下所有PNG图片
        image_files = sorted([f for f in os.listdir(load_path) if f.endswith('.png')])
        
        # 确保至少有4张图片
        if len(image_files) < 4:
            raise ValueError("目录中需要至少4张PNG图片")

        images = []
        for file in image_files[:5]:  # 取前5张图片
            img_path = os.path.join(load_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))  # 统一转为RGB格式
            images.append(img)

        # 验证所有图片尺寸一致
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        if len(set(heights)) != 1 or len(set(widths)) != 1:
            raise ValueError("所有图片必须具有相同的尺寸")

        # 创建空白画布
        h, w = heights[0], widths[0]
        num_images = len(images)
        gap_width = gap * (num_images - 1)  # 间隙总宽度
        canvas_width = w * num_images + gap_width  # 画布总宽度
        canvas = np.ones((h, canvas_width, 3), dtype=np.uint8) * 255  # 白色背景

        # 拼接图片
        x_offset = 0
        for img in images:
            canvas[:, x_offset:x_offset + w] = img
            x_offset += w + gap  # 添加间隙

        # 保存结果
        output_path = os.path.join(save_path, output_name)
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return output_path


@register_model
def tcformer_light(inchans=3, pretrained=False, **kwargs):
    model = TCFormer( in_chans=inchans,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], 
        Cmerge=False, FDPC_KNN=False, if_WSN=False,
        **kwargs)
    return model

@register_model
def tcformer(pretrained=False, Cmerge=False, FDPC_KNN=False, if_WSN=False, **kwargs):
    model = TCFormer(in_chans=3,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
       Cmerge=False, FDPC_KNN=False, if_WSN=False, **kwargs)
    return model


@register_model
def Ftcformer_light(in_chans=3, pretrained=False, Cmerge=True, FDPC_KNN=True, if_WSN=True, k_WSN=5, k=5, downsampling='CTM',
                **kwargs):
    model = TCFormer( in_chans=in_chans,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], 
        Cmerge=Cmerge, FDPC_KNN=FDPC_KNN, if_WSN=if_WSN, 
        k_WSN=k_WSN, k=k, downsampling=downsampling,
        **kwargs)
    return model


@register_model
def tcformer_large(pretrained=False, Cmerge=False, FDPC_KNN=False, if_WSN=False,**kwargs):
    model = TCFormer( in_chans=3,
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        Cmerge=False, FDPC_KNN=False, if_WSN=False,
        **kwargs)
    return model


from CNN_network import AlexNet as _AlexNet
@register_model
def AlexNet(num_classes=1000, in_chan=3, **kwargs):
    model = _AlexNet(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import AlexNet_FCTM as _AlexNet_FCTM
@register_model
def AlexNet_FCTM(num_classes=1000, in_chan=3, **kwargs):
    model = _AlexNet_FCTM(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import VGG19 as _VGG19
@register_model
def VGG19(num_classes=1000, in_chan=3, **kwargs):
    model = _VGG19(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import VGG19_FCTM as _VGG19_FCTM
@register_model
def VGG19_FCTM(num_classes=1000, in_chan=3, **kwargs):
    model = _VGG19_FCTM(num_classes=num_classes, in_chan=in_chan)
    return model


from CNN_network import ResNet18 as _ResNet18
@register_model
def ResNet18(num_classes=1000, in_chan=3, **kwargs):
    model = _ResNet18(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import ResNet18_FTCM as _ResNet18_FCTM
@register_model
def ResNet18_FCTM(num_classes=1000, in_chan=3, **kwargs):
    model = _ResNet18_FCTM(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import ResNeXt18 as _ResNeXt18
@register_model
def ResNeXt18(num_classes=1000, in_chan=3, **kwargs):
    model = _ResNeXt18(num_classes=num_classes, in_chan=in_chan)
    return model

from CNN_network import ResNeXt18_FCTM as _ResNeXt18_FCTM
@register_model
def ResNeXt18_FCTM(num_classes=1000, in_chan=3, **kwargs):
    model = _ResNeXt18_FCTM(num_classes=num_classes, in_chan=in_chan)
    return model



