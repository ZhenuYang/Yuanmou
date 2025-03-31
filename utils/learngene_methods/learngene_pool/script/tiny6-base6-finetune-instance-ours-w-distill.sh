CUDA_VISIBLE_DEVICES=7 python main.py --batch-size 256 --epoch 50 --lr 5e-4 --output_dir ./train_results/finetune_depth6/train_w_distill/instance_ours/ --blk_length 6 --init_stitch_mode ours --init_learngenepool_mode ours --ls_init True

# CUDA_VISIBLE_DEVICES=7 python main.py --batch-size 256 --epoch 50 --lr 5e-4 --output_dir ./train_results/finetune_depth6/test/instance_ours/ --blk_length 6 --init_stitch_mode ours --init_learngenepool_mode ours --eval --ls_init False
