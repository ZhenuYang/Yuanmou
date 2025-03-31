#!/bin/bash
#BSUB -J case
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=8:mode=exclusive_process:aff=yes"

module load anaconda3
source activate torch_shi

python -m torch.distributed.launch --nproc_per_node 8 --use_env distill.py --batch-size 256 --epochs 100 --num_workers 64 --lr 5e-4 --output_dir ./train_results/distill --teacher_model  deit_base_patch16_224 --student_model deit_small9_patch16_224 --data-path /seu_share/home/zhengfa_Test/user003/dataset/ILSVRC2012

# python distill.py --batch-size 256 --epochs 150 --num_workers 64 --data-path /seu_share/home/zhengfa_Test/user003/dataset/ILSVRC2012 --lr 1e-4 --output_dir ./train_results/distill
