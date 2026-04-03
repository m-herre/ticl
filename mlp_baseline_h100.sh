#!/bin/bash

#SBATCH --job-name=mlp_baseline_h100
#SBATCH --partition=gpu-vram-94gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=logs/mlp_baseline_h100_%j.out
#SBATCH --error=logs/mlp_baseline_h100_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

cd /work/mherre/ticl
mkdir -p logs runs models_diff

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-50}"
NUM_STEPS="${NUM_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
AGGREGATE_K_GRADIENTS="${AGGREGATE_K_GRADIENTS:-1}"
TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-true}"
RECOMPUTE_ATTN="${RECOMPUTE_ATTN:-false}"
ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-false}"
SAVE_EVERY="${SAVE_EVERY:-100}"
PROGRESS_BAR="${PROGRESS_BAR:-false}"
VALIDATE="${VALIDATE:-false}"
SEED_EVERYTHING="${SEED_EVERYTHING:-true}"

echo "Launching H100 MLP baseline"
echo "  epochs=${EPOCHS}"
echo "  num_steps=${NUM_STEPS}"
echo "  batch_size=${BATCH_SIZE}"
echo "  aggregate_k_gradients=${AGGREGATE_K_GRADIENTS}"
echo "  train_mixed_precision=${TRAIN_MIXED_PRECISION}"
echo "  recompute_attn=${RECOMPUTE_ATTN}"
echo "  adaptive_batch_size=${ADAPTIVE_BATCH_SIZE}"
echo "  validate=${VALIDATE}"
echo "  seed_everything=${SEED_EVERYTHING}"

cmd=(
    python ticl/fit_model.py mothernet
    -g "${GPU_ID}"
    --child-model mlp
    --epochs "${EPOCHS}"
    --num-steps "${NUM_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --aggregate_k_gradients "${AGGREGATE_K_GRADIENTS}"
    --train-mixed-precision "${TRAIN_MIXED_PRECISION}"
    --recompute-attn "${RECOMPUTE_ATTN}"
    --adaptive-batch-size "${ADAPTIVE_BATCH_SIZE}"
    --save-every "${SAVE_EVERY}"
    --progress-bar "${PROGRESS_BAR}"
    --validate "${VALIDATE}"
    --seed-everything "${SEED_EVERYTHING}"
)

if [[ "${USE_WANDB:-true}" == "true" ]]; then
    cmd+=(--use-wandb)
fi

"${cmd[@]}" "$@"

echo "Training completed at: $(date)"
