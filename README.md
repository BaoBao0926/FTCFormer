# FTCFormer: Fuzzy Token Clustering Transformer for Image Classification (ECAI'2025)

### Authors:
- **[Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)**, **Changyu Zeng**, **Yifan Wang**, **Zhengni Yang**, **Zimu Wang**, **Guangliang Cheng**, **Jun Qi**, **Wei Wang**

### NEWS:
- 2025.07.16: The code has been uploaded
- 2025.07.14：The paper has been accepted by **ECAI'2025**!
- 2025.05.16: The repository was created


# Abstract
Transformer-based deep neural networks have achieved remarkable success across various computer vision tasks, largely attributed to their long-range self-attention mechanism and scalability. However, most transformer architectures embed images into uniform, grid-based vision tokens, neglecting the underlying semantic meanings of image regions, resulting in suboptimal feature representations. To address this issue, we propose Fuzzy Token Clustering Transformer (FTCFormer), which incorporates a novel clustering-based downsampling module to dynamically generate vision tokens based on the semantic meanings instead of spatial positions. It allocates fewer tokens to less informative regions and more to represent semantically important regions, regardless of their spatial adjacency or shape irregularity. To further enhance feature extraction and representation, we propose a Density Peak Clustering-Fuzzy K-Nearest Neighbor (DPC-FKNN) mechanism for clustering center determination, a Spatial Connectivity Score (SCS) for token assignment, and a channel-wise merging (Cmerge) strategy for token merging. Extensive experiments on 32 datasets across diverse domains validate the effectiveness of FTCFormer on image classification, showing consistent improvements over the TCFormer baseline, achieving gains of improving 1.43% on five fine-grained datasets, 1.09% on six natural image datasets, 0.97% on three medical datasets and 0.55% on four remote sensing datasets.

# Archicture
<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/architecture.png" alt="architecture" width="700"/> 

# Visualization
<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/visualization%20figure.png" alt="visualization" width="700"/> 

# Datasets
<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/dataset.png" alt="datasets" width="500"/> 

# Experimental Results
<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/diff.png" alt="diff" width="700"/> 

<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/benchmark_three.png" alt="benchmark_three" width="700"/> 

<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/benchmark_imagenet.png" alt="benchmark_imagenet" width="500"/> 

# Ablation Study
<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/ablation.png" alt="ab" width="700"/> 

<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/ablation_K.png" alt="ab_k" width="700"/> 

<img src="https://github.com/BaoBao0926/FTCFormer/blob/main/Code/figure/ablation_downsampling.png" alt="ab_down" width="500"/> 

# Acknowledge
Thanks for the contribuction of [TCFomrer](https://github.com/zengwang430521/TCFormer).

