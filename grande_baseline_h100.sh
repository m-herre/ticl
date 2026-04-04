#!/bin/bash

#SBATCH --job-name=grande_baseline_h100
#SBATCH --partition=gpu-vram-32gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=logs/grande_baseline_h100_%j.out
#SBATCH --error=logs/grande_baseline_h100_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

cd /work/mherre/ticl
mkdir -p logs runs models_diff

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"
NUM_STEPS="${NUM_STEPS:-2048}"
# BATCH_SIZE="${BATCH_SIZE:-8}"
# AGGREGATE_K_GRADIENTS="${AGGREGATE_K_GRADIENTS:-1}"
TRAIN_MIXED_PRECISION="${TRAIN_MIXED_PRECISION:-true}"
RECOMPUTE_ATTN="${RECOMPUTE_ATTN:-false}"
# ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-false}"
# SAVE_EVERY="${SAVE_EVERY:-100}"
PROGRESS_BAR="${PROGRESS_BAR:-false}"
# VALIDATE="${VALIDATE:-false}"
SEED_EVERYTHING="${SEED_EVERYTHING:-true}"

TREE_DEPTH="${TREE_DEPTH:-4}"
N_ESTIMATORS="${N_ESTIMATORS:-64}"
SELECTED_VARIABLES="${SELECTED_VARIABLES:-16}"
DATA_SUBSET_FRACTION="${DATA_SUBSET_FRACTION:-1.0}"
BOOTSTRAP="${BOOTSTRAP:-false}"
GRANDE_DROPOUT="${GRANDE_DROPOUT:-0.0}"
MISSING_VALUES="${MISSING_VALUES:-true}"
# GRANDE_COMPILE="${GRANDE_COMPILE:-true}"

DIAGNOSTICS="${DIAGNOSTICS:-false}"
GRANDE_DIAGNOSTICS_LEVEL="${GRANDE_DIAGNOSTICS_LEVEL:-scalars_small_hists}"
GRANDE_DIAGNOSTICS_SEED="${GRANDE_DIAGNOSTICS_SEED:-0}"
GRANDE_DIAGNOSTICS_HIST_MAX_POINTS="${GRANDE_DIAGNOSTICS_HIST_MAX_POINTS:-2048}"
GRANDE_DIAGNOSTICS_GRADIENTS="${GRANDE_DIAGNOSTICS_GRADIENTS:-true}"

echo "Launching H100 GRANDE baseline"
echo "  epochs=${EPOCHS}"
echo "  num_steps=${NUM_STEPS}"
# echo "  batch_size=${BATCH_SIZE}"
# echo "  aggregate_k_gradients=${AGGREGATE_K_GRADIENTS}"
echo "  train_mixed_precision=${TRAIN_MIXED_PRECISION}"
echo "  recompute_attn=${RECOMPUTE_ATTN}"
# echo "  adaptive_batch_size=${ADAPTIVE_BATCH_SIZE}"
# echo "  validate=${VALIDATE}"
echo "  seed_everything=${SEED_EVERYTHING}"
echo "  tree_depth=${TREE_DEPTH}"
echo "  n_estimators=${N_ESTIMATORS}"
echo "  selected_variables=${SELECTED_VARIABLES}"
echo "  data_subset_fraction=${DATA_SUBSET_FRACTION}"
echo "  bootstrap=${BOOTSTRAP}"
echo "  grande_dropout=${GRANDE_DROPOUT}"
echo "  missing_values=${MISSING_VALUES}"
# echo "  grande_compile=${GRANDE_COMPILE}"
echo "  diagnostics=${DIAGNOSTICS}"
if [[ "${DIAGNOSTICS}" == "true" ]]; then
    echo "  grande_diagnostics_level=${GRANDE_DIAGNOSTICS_LEVEL}"
    echo "  grande_diagnostics_gradients=${GRANDE_DIAGNOSTICS_GRADIENTS}"
fi

cmd=(
    python ticl/fit_model.py mothernet
    -g "${GPU_ID}"
    --child-model grande
    --grande-decoder-variant factorized_stats
    --tree-depth "${TREE_DEPTH}"
    --n-estimators "${N_ESTIMATORS}"
    --selected-variables "${SELECTED_VARIABLES}"
    --data-subset-fraction "${DATA_SUBSET_FRACTION}"
    --bootstrap "${BOOTSTRAP}"
    --grande-dropout "${GRANDE_DROPOUT}"
    --missing-values "${MISSING_VALUES}"
    # --grande-compile "${GRANDE_COMPILE}"
    --epochs "${EPOCHS}"
    --num-steps "${NUM_STEPS}"
    # --batch-size "${BATCH_SIZE}"
    # --aggregate_k_gradients "${AGGREGATE_K_GRADIENTS}"
    --train-mixed-precision "${TRAIN_MIXED_PRECISION}"
    --recompute-attn "${RECOMPUTE_ATTN}"
    # --adaptive-batch-size "${ADAPTIVE_BATCH_SIZE}"
    # --save-every "${SAVE_EVERY}"
    --progress-bar "${PROGRESS_BAR}"
    # --validate "${VALIDATE}"
    --seed-everything "${SEED_EVERYTHING}"
)

if [[ "${USE_WANDB:-true}" == "true" ]]; then
    cmd+=(--use-wandb)
fi

if [[ "${DIAGNOSTICS}" == "true" ]]; then
    cmd+=(
        --grande-diagnostics true
        --grande-profile true
        --grande-diagnostics-gradients "${GRANDE_DIAGNOSTICS_GRADIENTS}"
        --grande-diagnostics-level "${GRANDE_DIAGNOSTICS_LEVEL}"
        --grande-diagnostics-seed "${GRANDE_DIAGNOSTICS_SEED}"
        --grande-diagnostics-hist-max-points "${GRANDE_DIAGNOSTICS_HIST_MAX_POINTS}"
    )
fi

"${cmd[@]}" "$@"

echo "Training completed at: $(date)"
