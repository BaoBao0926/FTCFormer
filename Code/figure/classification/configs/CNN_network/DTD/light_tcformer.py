cfg = dict(
    data_path='/dke/zcy/DTD', data_set='DTD', epochs=1000,
    model='tcformer_light',
    drop_path=0.1,
    clip_grad=None,
    output_dir='work_dirs/DTD/tcformer_light/1',
)


"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/DTD/light_tcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/DTD/light_tcformer.py
"""