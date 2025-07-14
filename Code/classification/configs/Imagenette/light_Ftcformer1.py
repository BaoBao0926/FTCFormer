cfg = dict(
    data_path='./datasets/Imagenette', data_set='Imagenette', epochs=300,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=60,
    lr=1.5e-3,
    k=5,k_WSN=5,
    # hyper parameter
    FDPC_KNN=False, if_WSN=False, Cmerge=False, # output_dir='work_dirs/Imagenette/light/tcformer/1',                    # here

    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/Imagenette/light/Fuzzy_tcformer/1',              # here        
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/Imagenette/light/Fuzzy_tcformer_WSN/1',          # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  output_dir='work_dirs/Imagenette/light/Fuzzy_tcformer_WSN_Cmerge/1',   # here
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/Imagenette/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/Imagenette/light_Ftcformer.py
"""