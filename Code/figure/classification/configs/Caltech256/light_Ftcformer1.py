cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/Caltech256', data_set='Caltech256', epochs=300,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=1,
    lr=1.5e-3,
    #k=5, k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, # output_dir='work_dirs/Caltech256/light/tcformer_1',                    # here 77.83
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/Caltech256/light/Fuzzy_tcformer/1',              # here 78.63        
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/Caltech256/light/Fuzzy_tcformer_WSN/1',          # here 78.79      
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  
    output_dir='work_dirs/Caltech256/light/Fuzzy_tcformer_WSN_Cmerge/1',  
    resume='/home/jitri7/muyi/TCFormer-master/classification/work_dirs/Caltech256/light/Fuzzy_tcformer_WSN_Cmerge/lr1.7e-3_75.72/best.pth'

    # downsampling='Convk33s2', output_dir='work_dirs/Caltech256/light/tcformer_Convk33s2/1',
    # downsampling='Convk22s2', output_dir='work_dirs/Caltech256/light/tcformer_Convk22s2/1',
    # downsampling='maxpooling', output_dir='work_dirs/Caltech256/light/tcformer_maxpooling/1',
    # downsampling='avgpooling', output_dir='work_dirs/Caltech256/light/tcformer_avgpooling/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/Caltech256/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/Caltech256/light_Ftcformer.py
"""