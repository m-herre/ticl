# Files

- `benchmark_node_gam_datasets.py` — Benchmarks multiple classifiers (LR, RF, XGBoost, EBM, GAMformer) on NODE-GAM datasets with cross-validated ROC-AUC scoring
- `case_studies.py` — Runs scaling analysis and shape function comparison between EBM and GAMformer on real datasets (MIMIC2/3, ADULT, SUPPORT2) and toy problems
- `concurvity.py` — Implements concurvity regularization (one-vs-rest and pairwise) for penalizing correlation between additive model components
- `critical_differences.py` — Draws critical difference diagrams using Wilcoxon signed-rank tests with Holm correction for comparing classifier rankings
- `demo_app.py` — Bokeh interactive web app for fitting EBM or GAMformer on the NSL-KDD intrusion detection dataset with slicing by categorical features
- `demo_plotting.py` — ipywidgets-based notebook demo for interactive shape function visualization with data filtering
- `fit_learning_curve.py` — Fits exponential decay curves to MLflow training loss histories and plots learning curves with extrapolation
- `imbalanced_data.py` — Evaluates GAMformer vs EBM robustness under class imbalance by sweeping minority-class ratios on synthetic data
- `node_gam_data.py` — Data loading and preprocessing for NODE-GAM benchmark datasets (MIMIC2, MIMIC3, ADULT, COMPAS, SUPPORT2, credit, bike, etc.)
- `noisy_data.py` — Evaluates GAMformer vs EBM robustness under label noise by sweeping flip probabilities on synthetic data
- `plot_shape_function.py` — Matplotlib plotting utilities for per-feature shape functions, comparing GAMformer and EBM outputs with data density overlays
- `tabular_evaluation.py` — Core evaluation framework for running models (transformer or baseline) on tabular datasets with train/test splitting and metric aggregation
- `tabular_metrics.py` — Metric functions (AUC, accuracy, cross-entropy, R2, RMSE, etc.) and per-framework scoring string mappings for sklearn, AutoGluon, XGBoost, CatBoost, etc.
- `utils.py` — Empty module placeholder

# Subdirectories

- `baselines/` — Baseline model implementations (MLP, ResNet, distillation) and sklearn-based evaluation harnesses for comparing against TabPFN/MotherNet
- `cd_plot_new/` — Updated critical difference and normalized improvement plot code using the autorank library for statistical comparisons
