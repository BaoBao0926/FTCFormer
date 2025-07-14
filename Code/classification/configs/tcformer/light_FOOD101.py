cfg = dict(
    model='tcformer_light', 
    data_path='/dke/zcy/FOOD101', data_set='FOOD101', epochs=500,  
    batch_size=60, lr=1.5e-3,
    drop_path=0.1,
    clip_grad=None,
    output_dir='work_dirs/tcformer/light_FOOD101_1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/tcformer/light_FOOD101.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/tcformer/light_FOOD101.py
"""