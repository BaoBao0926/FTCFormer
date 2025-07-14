cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/imagenet', 
    data_set='ImageNet', epochs=300,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    # clip_grad=0.1,
    clip_grad=None,
    batch_size=120,
    lr=1.0e-3,
    k=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/ImageNet/light/tcformer_1',                    # here
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/ImageNet/light/Fuzzy_tcformer_1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_1',          # her
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  
    output_dir='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_Cmerge/lr1.0_3', 
    resume='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_Cmerge/lr1.0_3/checkpoint.pth',

)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/ImageNet/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/ImageNet/light_Ftcformer.py
"""