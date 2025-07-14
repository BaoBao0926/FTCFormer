cfg = dict(
    data_path='/home/jitri7/muyi/TCFormer-master/Datasets/DTD', data_set='DTD', epochs=300,
    input_size=224,
    drop_path=0.1,
    clip_grad=None,
    batch_size=60,
    # lr=1.0e-3,
    # k=5,k_WSN=5,
    # hyper parameter
    # FDPC_KNN=False, if_WSN=False, Cmerge=False, output_dir='work_dirs/DTD/light/tcformer/1',                    # here
    # FDPC_KNN=True,  if_WSN=False, Cmerge=False, output_dir='work_dirs/DTD/light/Fuzzy_tcformer_1',              # here
    # FDPC_KNN=True,  if_WSN=True,  Cmerge=False, output_dir='work_dirs/DTD/light/Fuzzy_tcformer_WSN_1',          # here
    FDPC_KNN=True,  if_WSN=True,  Cmerge=True,  # output_dir='work_dirs/DTD/light/Fuzzy_tcformer_WSN_Cmerge/lr1.7',   # here

    # ---------------------------------------------------   AlexNet  -------------------------------------
    # model='AlexNet', output_dir = 'work_dirs/DTD/AlexNet/1',  
    # model='AlexNet_FCTM', output_dir = 'work_dirs/DTD/AlexNet_FCTM/1',  

    # --------------------------------------------------- VGG19  -----------------------------
    # model = 'VGG19', output_dir = 'work_dirs/DTD/VGG19/1',
    # model = 'VGG19_FCTM', output_dir = 'work_dirs/DTD/VGG19_FCTM/lr1.0e-3',

    # --------------------------------------------------- ResNet18 -----------------------------
    # model = 'ResNet18', output_dir = 'work_dirs/DTD/ResNet18/1',
    # model = 'ResNet18_FCTM', output_dir = 'work_dirs/DTD/ResNet18_FCTM/1',

    # --------------------------------------------------- ResNeXt18 -----------------------------
    # model = 'ResNeXt18', output_dir = 'work_dirs/DTD/ResNeXt18/1',
    # model = 'ResNeXt18_FCTM', output_dir = 'work_dirs/DTD/ResNeXt18_FCTM/1',
)

"""
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/CNN_network/DTD/light_Ftcformer.py
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/CNN_network/DTD/light_Ftcformer.py
"""