CUDA_VISIBLE_DEVICES=5 python main.py --batch-size 256 --epoch 50 --lr 5e-4 --output_dir ./train_results/finetune_depth6/train_w_distill/snnet_snnet/ --blk_length 6 --init_stitch_mode snnet --init_learngenepool_mode snnet --ls_init True