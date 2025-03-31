#!/bin/bash
#BSUB -J case
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"

module load anaconda3
source activate torch_shi


python scratch.py --batch-size 256 --epochs 150 --num_workers 64 --lr 5e-4 --output_dir ./train_results/scratch/ --data-path ../../dataset/ImageNet-1K --cfg_id 0 --blk_length 6