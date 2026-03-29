#!/bin/bash

#SBATCH --job-name=grande_hpo
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grande_hpo_%j.out
#SBATCH --error=logs/grande_hpo_%j.err

set -uo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Sweep dir: ${SWEEP_DIR}"
echo "Chunk ID: ${CHUNK_ID}"

source /home/mherre/miniconda3/etc/profile.d/conda.sh
conda activate ticl

cd /work/mherre/ticl
mkdir -p logs runs

CHUNK_FILE="${SWEEP_DIR}/chunk_${CHUNK_ID}.json"
if [ ! -f "${CHUNK_FILE}" ]; then
    echo "ERROR: Chunk file not found: ${CHUNK_FILE}"
    exit 1
fi

# Read sweep_id from metadata
SWEEP_ID=$(python -c "import json; print(json.load(open('${SWEEP_DIR}/sweep_meta.json'))['sweep_id'])")

# Count trials in this chunk
N_TRIALS=$(python -c "import json; print(len(json.load(open('${CHUNK_FILE}'))))")
echo "Running ${N_TRIALS} trials from ${CHUNK_FILE}"

# Run each trial sequentially
for TRIAL_IDX in $(seq 0 $((N_TRIALS - 1))); do
    TRIAL_ID=$(python -c "import json; print(json.load(open('${CHUNK_FILE}'))[${TRIAL_IDX}]['trial_id'])")
    echo ""
    echo "=========================================="
    echo "Starting trial ${TRIAL_ID} (${TRIAL_IDX}/${N_TRIALS}) at $(date)"
    echo "=========================================="

    # Convert trial config to CLI flags
    CLI_FLAGS=$(python -c "
import json, sys
sys.path.insert(0, '.')
from ticl.configs.grande_hpo import config_to_cli_flags
chunk = json.load(open('${CHUNK_FILE}'))
print(config_to_cli_flags(chunk[${TRIAL_IDX}]))
")

    # Set W&B tags and group via env vars
    export WANDB_TAGS="hpo,grande_hpo,${SWEEP_ID},trial_${TRIAL_ID}"
    export WANDB_RUN_GROUP="${SWEEP_ID}"

    echo "CLI flags: ${CLI_FLAGS}"
    echo "W&B group: ${SWEEP_ID}"
    echo "W&B tags: ${WANDB_TAGS}"

    # Run training — continue to next trial on failure
    eval python ticl/fit_model.py mothernet ${CLI_FLAGS} --use-wandb && \
        echo "Trial ${TRIAL_ID} completed successfully at $(date)" || \
        echo "WARNING: Trial ${TRIAL_ID} FAILED at $(date) — continuing to next trial"
done

echo ""
echo "All trials in chunk ${CHUNK_ID} finished at $(date)"
