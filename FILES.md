# Files

- `CLAUDE.md` — Instructions and architecture overview for Claude Code assistance
- `DECODER.md` — Conceptual guide to the currently supported decoder families and shared decoder readout modes
- `EXPERIMENTS.md` — Experiment log: hypotheses, configs, results, and conclusions
- `NOTICE.txt` — Copyright notice (University of Freiburg, original authors)
- `README.md` — Project README covering MotherNet, GAMformer, and TabFlex usage
- `azure-pipelines.yml` — Azure DevOps CI pipeline: conda setup, pytest, caching
- `continue_gradtree.sh` — SLURM job script for training/continuing GRANDE MotherNet runs
- `environment.yml` — Conda environment spec for the main `ticl` environment
- `grande.py` — Standalone GRANDE implementation (reference differentiable tree ensemble)
- `grande_diagnostics_metric_reference.md` — Reference for all tracked GRANDE diagnostics metrics, including per-prefix explanations and interpretation notes
- `grande_wandb_diagnostics_plan.md` — Implementation plan for per-epoch W&B diagnostics, initialization logging, and backbone/decoder gradient tracking for GRANDE MotherNet
- `mothernet_grande_decoder_analysis.md` — Analysis of why predicting GRANDE params is harder than MLP params
- `mothernet_train.sh` — SLURM job script for default MotherNet (MLP child) training
- `setup.cfg` — Python package config: package discovery and flake8 settings
- `setup.py` — Minimal setuptools entry point
- `submit_grande_depthwise_diagnostics.sh` — SLURM submit script for GRANDE MotherNet depthwise runs with W&B and per-epoch GRANDE diagnostics enabled
- `submit_grande_factorized_diagnostics.sh` — SLURM submit script for GRANDE MotherNet factorized runs with W&B and per-epoch GRANDE diagnostics enabled
- `tabflex_conda.yaml` — Conda environment spec for TabFlex (PyTorch 2.0, CUDA 11.8)
- `tabflex_conda_update.yaml` — Updated TabFlex conda environment with pinned versions

# Subdirectories

- `experiments/` — Jupyter notebooks for benchmarking and decoder analysis
- `log/` — Training log files from MotherNet/GRANDE runs
- `logs/` — SLURM job stdout/stderr output files
- `models_diff/` — Saved model checkpoints (.cpkt files)
- `scripts/` — Misc scripts and result CSVs
- `ticl/` — Main Python package: models, decoders, training, prediction, datasets, and tests
