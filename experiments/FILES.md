# Files

- `benchmark.ipynb` — Nested cross-validation benchmark comparing MotherNet variants (MLP, GRANDE depthwise, GRANDE factorized) against baselines (CART, Majority, Random) on OpenML-CC18 datasets, with calibration and per-class recall diagnostics.
- `decoder.ipynb` — Development notebook for testing and validating MLPModelDecoder, GradTreeDecoder, and the differentiable tree forward pass (shapes, gradients, straight-through estimators).
- `grande.ipynb` — Standalone GRANDE baseline notebook that trains the reference PyTorch GRANDE model on an OpenML dataset and reports accuracy, F1, and ROC-AUC.
- `test.ipynb` — End-to-end smoke tests of MotherNetClassifier on breast cancer data, including GradTree parameter extraction, decision tree visualization, feature importance analysis, and padding mask verification.
- `grande_diagnostics.ipynb` — Diagnostic notebook quantifying root causes of the GRANDE-vs-MLP MotherNet performance gap: feature selection entropy, seed sensitivity, split value distribution, estimator diversity, estimator weight uniformity, temperature calibration, and depth-wise gradient flow.
- `grande_wandb_analysis.ipynb` — Starter W&B analysis notebook for GRANDE diagnostics runs, with scalar trend plots, depth-wise collapse plots, histogram drift reconstruction, and optional gradient diagnostics visualization from init through later epochs.
