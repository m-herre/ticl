# Files

- `__init__.py` — Re-exports TabPFNClassifier, MotherNetClassifier, GAMformerClassifier, GAMformerRegressor, and EnsembleMeta
- `gamformer.py` — Sklearn-compatible GAMformer classifier and regressor that extract and predict with additive (GAM-style) models from a trained BAAM/additive MotherNet, with InterpretML-compatible global explanations
- `mothernet.py` — Sklearn-compatible MotherNet classifier supporting MLP, GradTree, and GRANDE child models, plus parameter extraction, standalone inference, ensemble meta-learning via label/feature shifts, and a MotherNet-initialized MLP fine-tuning classifier
- `tabflex.py` — TabFlex wrapper that selects among three pre-trained TabPFN variants based on dataset size and dimensionality
- `tabpfn.py` — Sklearn-compatible TabPFN classifier with ensemble predictions via class/feature permutations, preprocessing pipelines, and batched transformer inference
