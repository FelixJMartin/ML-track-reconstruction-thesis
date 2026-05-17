#!/bin/bash

CODE="/proj/uppmax2026-1-31/felix/code/Stage2_PhiFilter_cluster.py"
LOG_DIR="/proj/uppmax2026-1-31/felix/logs/grid_s2"
mkdir -p $LOG_DIR

for H1 in 256 512; do
for H2 in 128 256; do
for LR in 1e-3 5e-4; do
for PW in 10.0 30.0 68.0; do

    if [ "$H1" -le "$H2" ]; then continue; fi

    NAME="s2_h${H1}_${H2}_lr${LR}_pw${PW}"

    sbatch <<EOF
#!/bin/bash
#SBATCH -A uppmax2026-1-31
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -c 4
#SBATCH -t 03:30:00
#SBATCH -J ${NAME}
#SBATCH -o ${LOG_DIR}/${NAME}.out

ml CUDA
ml scikit-learn/1.6.1-gfbf-2024a

cd /proj/uppmax2026-1-31/felix/code
python Stage2_PhiFilter_cluster.py --h1 ${H1} --h2 ${H2} --lr ${LR} --pos_weight ${PW}
EOF

    echo "Submitted: ${NAME}"

done; done; done; done