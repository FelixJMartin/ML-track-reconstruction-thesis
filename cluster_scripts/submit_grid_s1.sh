#!/bin/bash

# Grid search over Stage 1 hyperparameters.
# Submits one SLURM job per configuration — all run in parallel.

for H1 in 256 512; do
for H2 in 128 256; do
for LR in 1e-3 5e-4 3e-4; do
for DR in 0.1 0.2; do

# skip H2 > H1 combinations
if [ "$H2" -gt "$H1" ]; then
    continue
fi

JOB_NAME="s1_h${H1}_${H2}_lr${LR}_dr${DR}"

sbatch <<EOF
#!/bin/bash
#SBATCH -A uppmax2026-1-31
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -J ${JOB_NAME}
#SBATCH -o /proj/uppmax2026-1-31/felix/models/grid_s1/log_${JOB_NAME}_%j.log

ml CUDA
ml scikit-learn/1.6.1-gfbf-2024a

cd /proj/uppmax2026-1-31/felix/code
python grid_search_s1_1000.py --h1 ${H1} --h2 ${H2} --lr ${LR} --dropout ${DR}
EOF

echo "Submitted: ${JOB_NAME}"

done
done
done
done