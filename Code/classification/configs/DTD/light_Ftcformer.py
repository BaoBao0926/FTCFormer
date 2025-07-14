cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/DTD', data_set='DTD', epochs=300,
    input_size=64,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=60,
    lr=1.7e-3,
    # k=5,k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/DTD/light/tcformer/1',                    # here
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/DTD/light/Fuzzy_tcformer_1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/DTD/light/Fuzzy_tcformer_WSN_1',          # here
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  output_dir='work_dirs/DTD/light/Fuzzy_tcformer_WSN_Cmerge/test',   # here

    decay_epochs=30, warmup_epochs=5, cooldown_epochs=10, patience_epochs=10, decay_rate=0.05
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/DTD/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/DTD/light_Ftcformer.py
"""