CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 \
 --use_env main.py \
 --config configs/ImageNet/light_Ftcformer.py \
 --output_dir 'work_dirs/ImageNet/light/Fuzzy_tcformer_WSN_Cmerge/lr1.0' \
 --lr 0.0010 

