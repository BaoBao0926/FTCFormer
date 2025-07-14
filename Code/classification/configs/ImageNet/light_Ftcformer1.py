cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/imagenet', data_set='ImageNet', epochs=300,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=120,
    lr=1.0e-3,
    k=5,
    # hyper parameter
    # here
    # this is for TCFormer
    # output_dir='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/ImageNet/light/tcformer/lr1.0e-3',
    # FDPC_KNN=False, if_WSN=False, Cmerge=False,
    # resume='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/ImageNet/light/tcformer/lr1.0e-3/checkpoint_161.pth'

    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/ImageNet/light/Fuzzy_tcformer_1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_1',          # here
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_Cmerge/1',   # here
    resume='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_Cmerge/lr1.0/weight/best_77.14.pth'

)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/ImageNet/light_Ftcformer1.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/ImageNet/light_Ftcformer1.py
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/ImageNet/light_Ftcformer1.py
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/ImageNet/light_Ftcformer1.py
"""
