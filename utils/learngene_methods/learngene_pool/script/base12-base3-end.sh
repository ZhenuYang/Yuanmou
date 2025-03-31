#!/bin/bash
#BSUB -J case
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=2:mode=exclusive_process:aff=yes"

module load anaconda3
source activate torch_shi

# python distill.py --batch-size 256 --epochs 100 --num_workers 64 --lr 5e-4 --output_dir ./train_results/distill --teacher_model  deit_base_patch16_224 --student_model deit_tiny9_patch16_224 --data-path ../../dataset/ImageNet-1K --loss_pos [end]


python -m torch.distributed.launch --nproc_per_node 2 --use_env distill.py --batch-size 256 --epochs 100 --num_workers 64 --lr 5e-4 --output_dir ./train_results/distill --teacher_model  deit_base_patch16_224 --student_model deit_base3_patch16_224 --data-path ../../dataset/ImageNet-1K --loss_pos [end]
