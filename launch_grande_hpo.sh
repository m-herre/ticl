#!/bin/bash
# Launch a GRANDE HPO random search sweep.
#
# Usage:
#   bash launch_grande_hpo.sh [N_TRIALS] [SEED] [N_JOBS]
#
# Examples:
#   bash launch_grande_hpo.sh 50          # 50 trials, seed=42, 10 jobs
#   bash launch_grande_hpo.sh 50 123      # 50 trials, seed=123, 10 jobs
#   bash launch_grande_hpo.sh 30 42 6     # 30 trials, seed=42, 6 jobs

set -euo pipefail

N_TRIALS="${1:-50}"
SEED="${2:-42}"
N_JOBS="${3:-10}"

echo "Launching GRANDE HPO sweep: ${N_TRIALS} trials, seed=${SEED}, ${N_JOBS} jobs"

cd /work/mherre/ticl
mkdir -p logs runs

# Step 1: Sample configs and create chunk files
OUTPUT=$(python ticl/configs/grande_hpo.py --n-trials "${N_TRIALS}" --seed "${SEED}" --n-jobs "${N_JOBS}")
echo "${OUTPUT}"

# Extract sweep directory from sampler output
SWEEP_DIR=$(echo "${OUTPUT}" | grep "^Sweep directory:" | awk '{print $3}')
SWEEP_ID=$(echo "${OUTPUT}" | grep "^Sweep ID:" | awk '{print $3}')

if [ -z "${SWEEP_DIR}" ]; then
    echo "ERROR: Could not determine sweep directory from sampler output"
    exit 1
fi

echo ""
echo "Sweep directory: ${SWEEP_DIR}"
echo "Sweep ID: ${SWEEP_ID}"
echo ""

# Step 2: Submit one SLURM job per chunk
JOB_IDS=()
for i in $(seq 0 $((N_JOBS - 1))); do
    CHUNK_FILE="${SWEEP_DIR}/chunk_${i}.json"
    # Skip empty chunks
    N_IN_CHUNK=$(python -c "import json; print(len(json.load(open('${CHUNK_FILE}'))))")
    if [ "${N_IN_CHUNK}" -eq 0 ]; then
        echo "Skipping chunk ${i} (empty)"
        continue
    fi

    JOB_ID=$(sbatch --export=ALL,SWEEP_DIR="${SWEEP_DIR}",CHUNK_ID="${i}" submit_grande_hpo.sh | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted chunk ${i} (${N_IN_CHUNK} trials) -> SLURM job ${JOB_ID}"
done

echo ""
echo "All ${#JOB_IDS[@]} jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "W&B group: ${SWEEP_ID}"

# Step 3: Record job IDs in sweep metadata
python -c "
import json
meta_path = '${SWEEP_DIR}/sweep_meta.json'
with open(meta_path) as f:
    meta = json.load(f)
meta['slurm_job_ids'] = '${JOB_IDS[*]}'.split()
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Updated {meta_path} with SLURM job IDs')
"
