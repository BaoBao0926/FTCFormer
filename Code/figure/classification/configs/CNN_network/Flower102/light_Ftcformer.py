cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets', data_set='FLOWER102', epochs=300,
    input_size=224,
    drop_path=0.1,
    clip_grad=0.1,
    batch_size=60,
    lr=1.0e-3,
    k=5, k_WSN=5,

    # ---------------------------------------------------   AlexNet  -------------------------------------
    # model='AlexNet', output_dir = 'work_dirs/Flower102/AlexNet/1',  
    # model='AlexNet_FCTM', output_dir = 'work_dirs/Flower102/AlexNet_FCTM/1',  

    # --------------------------------------------------- VGG19  -----------------------------
    # model = 'VGG19', output_dir = 'work_dirs/Flower102/VGG19/1',
    # model = 'VGG19_FCTM', output_dir = 'work_dirs/Flower102/VGG19_FCTM/lr1.0e-3',

    # --------------------------------------------------- ResNet18 -----------------------------
    # model = 'ResNet18', output_dir = 'work_dirs/Flower102/ResNet18/1',
    # model = 'ResNet18_FCTM', output_dir = 'work_dirs/Flower102/ResNet18_FCTM/1',

    # --------------------------------------------------- ResNeXt18 -----------------------------
    # model = 'ResNeXt18', output_dir = 'work_dirs/Flower102/ResNeXt18/1',
    # model = 'ResNeXt18_FCTM', output_dir = 'work_dirs/Flower102/ResNeXt18_FCTM/1',

    # -------------------------------------------------- hyper parameter -------------------------------------
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/Flower102/light/tcformer_1',                    

    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/Flower102/light/Fuzzy_tcformer/1',              
    # FDPC_KNN=False, if_WSN=True,  Cmerge=False, output_dir='work_dirs/Flower102/light/tcformer_WSN/1',                
    # FDPC_KNN=False, if_WSN=False, Cmerge=True,  output_dir='work_dirs/Flower102/light/tcformer_Cmerge/1',             
    
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/Flower102/light/Fuzzy_tcformer_WSN/1',          
    # FDPC_KNN=True,  if_WSN=False, Cmerge=True,  output_dir='work_dirs/Flower102/light/Fuzzy_tcformer_Cmerge_1',       
    FDPC_KNN=False, if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/Flower102/light/tcformer_WSN_Cmerge_1',           

    # FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/Flower102/light/Fuzzy_tcformer_WSN_Cmerge_1',   

    # downsampling='Convk33s2', output_dir='work_dirs/Flower102/light/tcformer_Convk33s2/1',
    # downsampling='Convk22s2', output_dir='work_dirs/Flower102/light/tcformer_Convk22s2/1',
    # downsampling='maxpooling', output_dir='work_dirs/Flower102/light/tcformer_maxpooling/1',
    # downsampling='avgpooling', output_dir='work_dirs/Flower102/light/tcformer_avgpooling/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/CNN_network/Flower102/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/CNN_network/Flower102/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/CNN_network/Flower102/light_Ftcformer.py
"""