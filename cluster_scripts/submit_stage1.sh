#!/bin/bash
#SBATCH -A uppmax2026-1-31
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -J stage1_train
#SBATCH -o /proj/uppmax2026-1-31/felix/models/stage1_output_%j.log

ml CUDA
ml scikit-learn/1.6.1-gfbf-2024a

cd /proj/uppmax2026-1-31/felix/code
python stage1_uppmax.py