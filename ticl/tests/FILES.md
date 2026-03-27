# Files

- `__init__.py` — Empty package init.
- `test_categorical_embedding.py` — Tests `_determine_is_categorical` marks the correct features as categorical in GAMformer input encoding.
- `test_dataloader.py` — Tests `get_dataloader` with various prior types, batch sizes, feature counts, NaN injection, and uninformative features.
- `test_fit_model_parsing.py` — Tests that `fit_model.main(['--help'])` exits cleanly with code 0.
- `test_follow_sklearn_interface.py` — Tests sklearn-compatible fit/predict/score/pickle for TabPFN, MotherNet, GAMformer (classification and regression), BAAM, and DistilledTabPFNMLP.
- `test_get_gradients.py` — Tests that TabPFN supports gradient-based optimization of input data by verifying monotone loss decrease over gradient steps.
- `test_grande_diagnostics.py` — Tests deterministic GRANDE diagnostic snapshotting plus forward-only and gradient-enabled diagnostics collection during training.
- `test_grande_core.py` — Tests GRANDE core components: context building, tree routing, decoder output shapes, factorized decoder properties, temperature annealing, feature statistics, and extract-then-predict round-trip.
- `test_learning_rate_schedulers.py` — Tests cosine, exponential, and constant LR schedules respect min_lr bounds, and validates LR decay during short training runs.
- `test_load_module_only_inference.py` — Tests that TabPFN loaded in inference-only mode produces identical predictions to a normally loaded model on OpenML data.
- `test_model_builder.py` — Tests `load_model` backfills missing GRANDE buffers in legacy checkpoints and raises on missing required parameters.
- `test_mothernet_gradtree.py` — Tests that MotherNetClassifier with GradTree child model correctly undoes label offset when returning prediction probabilities.
- `test_standardization.py` — Tests that the refactored `remove_outliers` produces identical results to the original implementation across OpenML datasets.
- `test_utils.py` — Tests `get_wandb_run_string` preserves short names and deterministically truncates long names to 128 characters.

# Subdirectories

- `baselines/` — Tests for baseline model training (TorchMLP, ResNet).
- `models/` — Tests for model utility functions (data binning).
- `prediction/` — Tests for prediction/inference paths of MotherNet, TabPFN, TabFlex, and MotherNetInitMLP.
- `priors/` — Tests for synthetic data priors (boolean, classification, MLP, step function, differentiable hyperparameters).
- `training/` — Tests for end-to-end training loops of all model architectures (additive, BAAM, BiAttentionTabPFN, MotherNet, Perceiver, TabFlex, TabPFN).
