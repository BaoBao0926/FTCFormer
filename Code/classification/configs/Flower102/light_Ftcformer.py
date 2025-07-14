cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets', data_set='FLOWER102', epochs=14,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    # clip_grad=0.1,
    batch_size=120,
    lr=1.5e-3,
    k=5, k_WSN=5,
    # hyper parameter
    FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/Flower102/light/tcformer/test',         
    # resume='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/Flower102/light/tcformer/toy_clip/1/checkpoint_9.pth',
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/Flower102/light/Fuzzy_tcformer/1',                      
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/Flower102/light/Fuzzy_tcformer_WSN/1',                 
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  output_dir='work_dirs/Flower102/light/Fuzzy_tcformer_WSN_Cmerge_1',   

    warmup_epochs=0,

#     parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
#                    help='warmup learning rate (default: 1e-6)')
# parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
#                    help='lower lr bound for cyclic schedulers (default: 1e-5)')
# parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
#                    help='epochs to warmup LR (default: 5)')

    # downsampling='Convk33s2', output_dir='work_dirs/Flower102/light/tcformer_Convk33s2/1',
    # downsampling='Convk22s2', output_dir='work_dirs/Flower102/light/tcformer_Convk22s2/1',
    # downsampling='maxpooling', output_dir='work_dirs/Flower102/light/tcformer_maxpooling/1',
    # downsampling='avgpooling', output_dir='work_dirs/Flower102/light/tcformer_avgpooling/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/Flower102/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/Flower102/light_Ftcformer.py
"""