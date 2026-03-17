#!/usr/bin/env python3
"""Run a single GRANDE autoresearch experiment.

Reads hyperparameters from config.py, launches ticl/fit_model.py via
subprocess, and prints a grep-friendly summary.  Must be run from the
ticl project root (/work/mherre/ticl).
"""

import re
import subprocess
import sys
import time
from pathlib import Path

# ── import config ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TIMEOUT_SECONDS = 20 * 60  # 20 minutes
LOG_FILE = Path(__file__).resolve().parent / "run.log"


def build_command():
    """Build the CLI command list from config constants."""
    cmd = [
        sys.executable, "ticl/fit_model.py", "mothernet",
        "--child-model", "grande",
        "--tree-depth", str(config.TREE_DEPTH),
        "--n-estimators", str(config.N_ESTIMATORS),
        "--selected-variables", str(config.SELECTED_VARIABLES),
        "--data-subset-fraction", str(config.DATA_SUBSET_FRACTION),
        "--bootstrap", str(config.BOOTSTRAP),
        "--grande-dropout", str(config.GRANDE_DROPOUT),
        "--missing-values", str(config.MISSING_VALUES),
        "--emsize", str(config.EMSIZE),
        "--nlayers", str(config.NLAYERS),
        "--decoder-type", config.DECODER_TYPE,
        "--decoder-embed-dim", str(config.DECODER_EMBED_DIM),
        "--learning-rate", str(config.LEARNING_RATE),
        "--weight-decay", str(config.WEIGHT_DECAY),
        "--warmup-epochs", str(config.WARMUP_EPOCHS),
        "--batch-size", str(config.BATCH_SIZE),
        "--num-steps", str(config.NUM_STEPS),
        "--stop-after-epochs", str(config.EPOCHS),
        "--save-every", str(config.SAVE_EVERY),
        "--validate", "True",
        "--progress-bar", "False",
        "--warm-start-from", config.WARM_START_FROM,
    ]
    return cmd


def parse_log(log_text):
    """Extract metrics from training log output."""
    val_score = None
    train_loss = None

    # val_score: last occurrence of "Validation score: <float>"
    for m in re.finditer(r"Validation score:\s*([\d.]+)", log_text):
        val_score = float(m.group(1))

    # train_loss: last occurrence of "mean loss <float>"
    for m in re.finditer(r"mean loss\s+([\d.]+)", log_text):
        train_loss = float(m.group(1))

    return val_score, train_loss


def main():
    cmd = build_command()
    print(f"Running: {' '.join(cmd)}", flush=True)
    print(f"Logging to: {LOG_FILE}", flush=True)

    start = time.time()
    try:
        with open(LOG_FILE, "w") as log_fh:
            result = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
            )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: experiment exceeded {TIMEOUT_SECONDS}s", file=sys.stderr)
        returncode = -1

    elapsed = time.time() - start

    # Read log
    log_text = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    val_score, train_loss = parse_log(log_text)

    # Print summary
    print("---")
    if val_score is not None:
        print(f"val_score:        {val_score:.4f}")
    else:
        print("val_score:        CRASH")
    if train_loss is not None:
        print(f"train_loss:       {train_loss:.4f}")
    else:
        print("train_loss:       N/A")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"exit_code:        {returncode}")

    sys.exit(0 if val_score is not None else 1)


if __name__ == "__main__":
    main()
