#!/bin/bash
#SBATCH -A uppmax2026-1-31
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J stage2_8500
#SBATCH -o /proj/uppmax2026-1-31/felix/logs/stage2_8500_%j.out

module load python/3.11
source /proj/mixed-precision/mixed-precision/felix/venv/bin/activate

python /proj/mixed-precision/mixed-precision/felix/code/grid_search_s2_8500.py