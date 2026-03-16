#!/bin/bash

#SBATCH --job-name=gradtree_mothernet
#SBATCH --partition=gpu-vram-94gb
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --output=logs/gradtree_%j.out
#SBATCH --error=logs/gradtree_%j.err
#SBATCH --exclude=dws-09,dws-10

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load conda environment
source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

# Change to project directory
cd /work/mherre/ticl

# Create logs directory if it doesn't exist
mkdir -p logs

# Set checkpoint path (use the latest one)
# CHECKPOINT="/work/mherre/mothernet/ticl/models_diff/mn_childmodelgradtree_nestimators32_n2048_treedepth3_U1_12_03_2025_14_18_06_epoch_30.cpkt"

# echo "Continuing training from: $CHECKPOINT"

# # Continue training from checkpoint with additional options
# python ticl/fit_model.py mothernet \
#     --child-model gradtree \
#     --progress-bar False \
#     --use-wandb \
#     --tree-depth 3 \
#     --n-estimators 32 \
#     --continue-run \
#     --warm-start-from $CHECKPOINT \

python ticl/fit_model.py mothernet \
    --child-model gradtree \
    --progress-bar False \
    --use-wandb \
    --tree-depth 4 \
    --n-estimators 64 \
    --warmup-epochs 1 \
    --num-steps 2048



echo "Training completed at: $(date)"