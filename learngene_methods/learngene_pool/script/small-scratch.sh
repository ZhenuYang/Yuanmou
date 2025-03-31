#!/bin/bash
#BSUB -J case
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=4:mode=exclusive_process:aff=yes"

module load anaconda3
source activate torch_shi


python -m torch.distributed.launch --nproc_per_node 4 --master_port 15551 --use_env scratch.py --batch-size 256 --epochs 150 --num_workers 64 --lr 5e-4 --output_dir ./train_results/scratch/small_v2 --data-path ../../dataset/ImageNet-1K --resume ./train_results/scratch/small/[2,2,2]/pretrain_checkpoint.pth