cfg = dict(
    data_path='./datasets/RESISC45', data_set='RESISC45', epochs=300,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=1,
    # lr=1.8e-3,
    k=5,k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, # output_dir='work_dirs/RESISC45/light/tcformer/1',                    # here

    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/RESISC45/light/Fuzzy_tcformer/1',              # here        
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/RESISC45/light/Fuzzy_tcformer_WSN/1',          # here
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True, # output_dir='work_dirs/RESISC45/light/Fuzzy_tcformer_WSN_Cmerge/1',   # here
    resume = '/home/cheng/muyi/TCFormer/classification/work_dirs/RESISC45/light/Fuzzy_tcformer_WSN_Cmerge/lr1.5_96.62/best.pth'
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/RESISC45/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/RESISC45/light_Ftcformer.py
"""