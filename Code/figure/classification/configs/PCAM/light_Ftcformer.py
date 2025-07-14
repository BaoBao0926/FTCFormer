cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/PCAM', data_set='PCAM', epochs=100,
    input_size=96,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=480,
    # lr=9.0e-3,
    k=5,k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/PCAM/light/tcformer_1',                    # here
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/PCAM/light/Fuzzy_tcformer_1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/PCAM/light/Fuzzy_tcformer_WSN_1',          # here        
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/PCAM/light/Fuzzy_tcformer_WSN_Cmerge/1',   # here
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/PCAM/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/PCAM/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/PCAM/light_Ftcformer.py
"""