cfg = dict(
    data_path='/dke/zcy/FLOWER102', data_set='FLOWER102', epochs=1000,
    model='tcformer_light',
    drop_path=0.1,
    clip_grad=None,
    output_dir='work_dirs/Flower102/tcformer_light/1',
)


"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/Flower102/light_tcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/Flower102/light_tcformer.py
"""