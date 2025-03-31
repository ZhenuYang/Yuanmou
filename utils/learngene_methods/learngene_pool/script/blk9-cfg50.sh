python -m torch.distributed.launch --nproc_per_node 2 --use_env scratch.py --batch-size 256 --epochs 150 --num_workers 64 --lr 5e-4 --output_dir ./train_results/all_results/scratch/blk_length6/cfg_id50  --data-path /root/data2/ImageNet1K/ --cfg_id 50 --blk_length 9