cfg = dict(
    data_path='/dke/zcy/FOOD101', data_set='FOOD101', epochs=500,
    model='tcformer_light',
    drop_path=0.1,
    clip_grad=None,
    output_dir='work_dirs/Food101/tcformer_light/1',
)


"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/Food101/light_tcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/Food101/light_tcformer.py
"""