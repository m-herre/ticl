#!/bin/bash

#SBATCH --job-name=grande_factorized
#SBATCH --partition=gpu-vram-32gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=logs/grande_factorized_%j.out
#SBATCH --error=logs/grande_factorized_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

cd /work/mherre/ticl
mkdir -p logs runs

EPOCHS="${EPOCHS:-50}"
NUM_STEPS="${NUM_STEPS:-2048}"
TREE_DEPTH="${TREE_DEPTH:-4}"
N_ESTIMATORS="${N_ESTIMATORS:-64}"
SELECTED_VARIABLES="${SELECTED_VARIABLES:-16}"
PROGRESS_BAR="${PROGRESS_BAR:-False}"
GRANDE_DIAGNOSTICS_LEVEL="${GRANDE_DIAGNOSTICS_LEVEL:-scalars_small_hists}"
GRANDE_DIAGNOSTICS_SEED="${GRANDE_DIAGNOSTICS_SEED:-0}"
GRANDE_DIAGNOSTICS_HIST_MAX_POINTS="${GRANDE_DIAGNOSTICS_HIST_MAX_POINTS:-2048}"
GRANDE_DIAGNOSTICS_GRADIENTS="${GRANDE_DIAGNOSTICS_GRADIENTS:-True}"

echo "Launching GRANDE factorized diagnostics run"
echo "  epochs=${EPOCHS}"
echo "  num_steps=${NUM_STEPS}"
echo "  tree_depth=${TREE_DEPTH}"
echo "  n_estimators=${N_ESTIMATORS}"
echo "  selected_variables=${SELECTED_VARIABLES}"
echo "  grande_diagnostics_level=${GRANDE_DIAGNOSTICS_LEVEL}"
echo "  grande_diagnostics_gradients=${GRANDE_DIAGNOSTICS_GRADIENTS}"

python ticl/fit_model.py mothernet \
    --child-model grande \
    --grande-decoder-variant factorized_stats \
    --tree-depth "${TREE_DEPTH}" \
    --n-estimators "${N_ESTIMATORS}" \
    --selected-variables "${SELECTED_VARIABLES}" \
    --data-subset-fraction 1.0 \
    --bootstrap False \
    --grande-dropout 0.0 \
    --missing-values True \
    --epochs "${EPOCHS}" \
    --num-steps "${NUM_STEPS}" \
    --progress-bar "${PROGRESS_BAR}" \
    --use-wandb \
    --grande-diagnostics True \
    --grande-diagnostics-gradients "${GRANDE_DIAGNOSTICS_GRADIENTS}" \
    --grande-diagnostics-level "${GRANDE_DIAGNOSTICS_LEVEL}" \
    --grande-diagnostics-seed "${GRANDE_DIAGNOSTICS_SEED}" \
    --grande-diagnostics-hist-max-points "${GRANDE_DIAGNOSTICS_HIST_MAX_POINTS}" \
    "$@"

echo "Training completed at: $(date)"
