#!/bin/bash
# submit_grid_s1.sh

CODE="/proj/uppmax2026-1-31/felix/code/Stage1_ConeFilter_cluster.py"
LOG_DIR="/proj/uppmax2026-1-31/felix/logs/grid_s1"
mkdir -p $LOG_DIR

for H1 in 256 512; do
for H2 in 128 256; do
for LR in 1e-3 5e-4; do
for DR in 0.1 0.2; do

    if [ "$H1" -le "$H2" ]; then continue; fi

    NAME="s1_h${H1}_${H2}_lr${LR}_dr${DR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH -A uppmax2026-1-31
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 4
#SBATCH -t 02:30:00
#SBATCH -J ${NAME}
#SBATCH -o ${LOG_DIR}/${NAME}.out

ml CUDA
ml scikit-learn/1.6.1-gfbf-2024a

cd /proj/uppmax2026-1-31/felix/code
python Stage1_ConeFilter_cluster.py --h1 ${H1} --h2 ${H2} --lr ${LR} --dropout ${DR}
EOF

    echo "Submitted: ${NAME}"

done; done; done; done
