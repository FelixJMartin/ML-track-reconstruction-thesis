#!/bin/bash
# launch_stage2_grid.sh
# Submits all 24 Stage 2 grid search jobs in parallel.
# Run from: /proj/uppmax2026-1-31/felix/code/
# Usage: bash launch_stage2_grid.sh

mkdir -p /proj/uppmax2026-1-31/felix/models/grid_s2

H1_LIST="256 512"
H2_LIST="128 256"
LR_LIST="1e-3 5e-4 3e-4"
DR_LIST="0.1 0.2"

COUNT=0
for H1 in $H1_LIST; do
for H2 in $H2_LIST; do
for LR in $LR_LIST; do
for DR in $DR_LIST; do
    NAME="s2_h${H1}_${H2}_lr${LR}_dr${DR}"
    sbatch \
        --job-name="$NAME" \
        --output="/proj/uppmax2026-1-31/felix/models/grid_s2/log_${NAME}_%j.log" \
        /proj/uppmax2026-1-31/felix/code/submit_stage2_grid.sh $H1 $H2 $LR $DR
    COUNT=$((COUNT + 1))
done
done
done
done

echo "Submitted $COUNT jobs."