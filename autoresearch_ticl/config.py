"""Editable hyperparameters for autoresearch GRANDE experiments.

The autonomous agent modifies this file between runs. run_experiment.py
reads these constants to build the training CLI command.
"""

# ── GRANDE Tree Architecture ──────────────────────────────────────────
TREE_DEPTH = 4
N_ESTIMATORS = 64
SELECTED_VARIABLES = 16

# ── GRANDE Training ───────────────────────────────────────────────────
DATA_SUBSET_FRACTION = 1.0
BOOTSTRAP = False
GRANDE_DROPOUT = 0.0
MISSING_VALUES = True

# ── Transformer Backbone ─────────────────────────────────────────────
EMSIZE = 512
NLAYERS = 12

# ── Decoder ───────────────────────────────────────────────────────────
DECODER_TYPE = "class_average"
DECODER_EMBED_DIM = 1024

# ── Optimizer ─────────────────────────────────────────────────────────
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.0
WARMUP_EPOCHS = 1
BATCH_SIZE = 8

# ── Experiment Budget (fixed — agent should NOT change these) ─────────
NUM_STEPS = 256
EPOCHS = 5
SAVE_EVERY = 5

# ── Baseline checkpoint ──────────────────────────────────────────────
WARM_START_FROM = "/work/mherre/ticl/models_diff/mn_childmodelgrande_nestimators64_n2048_treedepth4_continue_03_17_2026_09_24_09_epoch_70.cpkt"
