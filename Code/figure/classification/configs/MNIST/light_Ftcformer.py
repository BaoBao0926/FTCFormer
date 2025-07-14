cfg = dict(
    data_path='/dke/zcy/MNIST', data_set='MNIST', epochs=200,
    input_size=64,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=1, in_chans=1,
    lr=1.7e-3,
    k=5, k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/MNIST/light/tcformer/2',                    # here 
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/MNIST/light/Fuzzy_tcformer/1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/MNIST/light/Fuzzy_tcformer_WSN/1',          # here
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/MNIST/light/Fuzzy_tcformer_WSN_Cmerge/1',   # here 
    resume='/home/zengchangyu/muyi/TCFormer-master/classification/work_dirs/MNIST/light/Fuzzy_tcformer_WSN_Cmerge/99.79/best.pth'

    # downsampling='Convk33s2', output_dir='work_dirs/MNIST/light/tcformer_Convk33s2/1',
    # downsampling='Convk22s2', output_dir='work_dirs/MNIST/light/tcformer_Convk22s2/1',
    # downsampling='maxpooling', output_dir='work_dirs/MNIST/light/tcformer_maxpooling/1',
    # downsampling='avgpooling', output_dir='work_dirs/MNIST/light/tcformer_avgpooling/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/MNIST/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/MNIST/light_Ftcformer.py
"""