cfg = dict(
    data_path='/dke/zcy/StanfordCar', data_set='StanfordCar', epochs=1000,
    input_size=224,
    model='Ftcformer_light',
    drop_path=0.1,
    clip_grad=None,
    batch_size=60,
    lr=1.5e-3,
    k=5, k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/StanfordCar/light/tcformer_1',                    # here 77.83

    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/StanfordCar/light/Fuzzy_tcformer/1',              # here 78.63
    # FDPC_KNN=False, if_WSN=True,  Cmerge=False, output_dir='work_dirs/StanfordCar/light/tcformer_WSN/1',                # 00
    # FDPC_KNN=False, if_WSN=False, Cmerge=True,  output_dir='work_dirs/StanfordCar/light/tcformer_Cmerge/1',             
    
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/StanfordCar/light/Fuzzy_tcformer_WSN/1',          # here 78.79
    # FDPC_KNN=True,  if_WSN=False, Cmerge=True,  output_dir='work_dirs/StanfordCar/light/Fuzzy_tcformer_Cmerge_1',       
    # FDPC_KNN=False, if_WSN=True,  Cmerge=True,  output_dir='work_dirs/StanfordCar/light/tcformer_WSN_Cmerge_1',           

    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  output_dir='work_dirs/StanfordCar/light/Fuzzy_tcformer_WSN_Cmerge_1',   # here 79.12

    # downsampling='Convk33s2', output_dir='work_dirs/StanfordCar/light/tcformer_Convk33s2/1',
    # downsampling='Convk22s2', output_dir='work_dirs/StanfordCar/light/tcformer_Convk22s2/1',
    # downsampling='maxpooling', output_dir='work_dirs/StanfordCar/light/tcformer_maxpooling/1',
    # downsampling='avgpooling', output_dir='work_dirs/StanfordCar/light/tcformer_avgpooling/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/StanfordCar/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/StanfordCar/light_Ftcformer.py
"""