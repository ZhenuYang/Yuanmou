#!/bin/bash


# python distill.py --batch-size 256 --epochs 100 --num_workers 64 --lr 5e-4 --output_dir ./train_results/distill --teacher_model  deit_base_patch16_224 --student_model deit_tiny9_patch16_224 --data-path ../../dataset/ImageNet-1K --loss_pos [end]


python -m torch.distributed.launch --nproc_per_node 2 --use_env distill.py --batch-size 256 --epochs 100 --num_workers 64 --lr 5e-4 --output_dir ./train_results/distill --teacher_model  deit_base_patch16_224 --student_model deit_tiny6_patch16_224 --data-path /root/data2/ImageNet1K --loss_pos [end] --alpha 0.1
