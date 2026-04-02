#!/bin/bash

#SBATCH --job-name=grande_long
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=logs/grande_long_%j.out
#SBATCH --error=logs/grande_long_%j.err

set -uo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

cd /work/mherre/ticl
mkdir -p logs

# W&B tagging
export WANDB_TAGS="grande_long,grande_hpo_best"
export WANDB_RUN_GROUP="grande_long"

python ticl/fit_model.py mothernet \
    --child-model grande \
    --grande-decoder-variant factorized_stats \
    --tree-depth 5 \
    --n-estimators 128 \
    --selected-variables 32 \
    --data-subset-fraction 0.5534 \
    --grande-dropout 0.05 \
    --grande-diversity-loss-weight 0.01 \
    --grande-output-init default \
    --decoder-hidden-size 1024 \
    --decoder-hidden-layers 1 \
    --decoder-embed-dim 512 \
    --emsize 1024 \
    --nlayers 6 \
    --learning-rate 0.0001479 \
    --weight-decay 0.01 \
    --batch-size 16 \
    --num-steps 2048 \
    --missing-values True \
    --bootstrap False \
    --epochs 2000 \
    --validate True \
    --save-every 10 \
    --progress-bar False \
    --grande-diagnostics True \
    --grande-diagnostics-gradients True \
    --grande-diagnostics-level scalars_small_hists \
    --grande-diagnostics-seed 0 \
    --grande-diagnostics-hist-max-points 2048 \
    --use-wandb

echo "Finished at $(date)"
