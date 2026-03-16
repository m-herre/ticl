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

python ticl/fit_model.py mothernet \
    --progress-bar False \
    --use-wandb \

echo "Training completed at: $(date)"