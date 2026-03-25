# Files

- `baseline_prediction_interface.py` — Generic prediction interface that splits eval data into train/test and dispatches to a given metric function for baseline models
- `distill_mlp.py` — Distilled TabPFN-to-MLP classifier that trains a TorchMLP on soft probability targets produced by TabPFN
- `evaluate_baselines_sklearn.py` — Cross-validates multiple sklearn classifiers (LogReg, KNN, HGB, RF, TabPFN, MLP, distilled TabPFN) on OpenML CC datasets
- `evaluate_mlps_sklearn.py` — Factory functions for building sklearn pipelines with MotherNet, TabPFN, distilled TabPFN, and TorchMLP variants at different sizes
- `resnet.py` — ResNet for tabular data (from RTDL) with a sklearn-compatible classifier wrapper using TorchModelTrainer
- `tabular_baselines.py` — Hyperparameter-tuned baselines (KNN, GP, CatBoost, XGBoost, LightGBM, sklearn MLP, etc.) using Hyperopt search with cross-validation
- `torch_mlp.py` — PyTorch MLP module and sklearn-compatible TorchModelTrainer base class for training, predicting, and evaluating neural network classifiers
