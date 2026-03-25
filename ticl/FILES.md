# Files

- `__init__.py` — Package entry point; exports `TabPFNClassifier`.
- `cli_parsing.py` — Argparse definitions for all model types and their hyperparameters.
- `config_utils.py` — Dict utilities: flatten, compare, merge, update, and `str2bool` helper.
- `conftest.py` — Pytest session fixture that limits torch threads to 1.
- `dataloader.py` — `PriorDataLoader` that wraps synthetic priors into a training data stream.
- `distributions.py` — Hyperparameter sampling distributions (beta, gamma, uniform, meta-choices, etc.).
- `environment.py` — Weights & Biases project/entity configuration constants.
- `fit_model.py` — CLI entry point: parses args, sets up logging, and launches training.
- `model_builder.py` — Model construction, checkpoint loading, loss selection, and training dispatch.
- `model_configs.py` — Default config dicts for each model type (MotherNet, TabPFN, additive, etc.).
- `model_eval.py` — Script to evaluate models on OpenML benchmark datasets.
- `profiling_prediction.py` — Script to profile MotherNet inference on validation datasets.
- `testing_utils.py` — Shared test helpers: default CLI args, iris/regression smoke-test functions.
- `train.py` — Training loop: epoch iteration, optimizer, LR scheduling, and logging.
- `utils.py` — General utilities: device init, model string generation, NaN-aware stats, validation.

# Subdirectories

- `analysis/` — Exploratory ICL analysis notebooks.
- `configs/` — Per-model-type configuration scripts for experiment sweeps.
- `datasets/` — OpenML dataset loading, caching, and synthetic dataset evaluation.
- `evaluation/` — Benchmark evaluation, metrics, baselines, and plotting utilities.
- `models/` — Neural network architectures: MotherNet, TabPFN, decoders, encoders, GRANDE core.
- `notebooks/` — Jupyter notebooks for result visualization and comparison.
- `prediction/` — Sklearn-compatible wrappers for inference (TabPFN, MotherNet, GAMformer).
- `priors/` — Synthetic data priors: MLP, GP, boolean conjunctions, step functions.
- `tests/` — Unit and integration tests for models, training, and prediction.
