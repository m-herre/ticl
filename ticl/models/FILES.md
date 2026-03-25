# Files

- `__init__.py` — Empty init file for the models package.
- `biattention_tabpfn.py` — TabPFN variant using bi-attention (cross-feature + cross-sample) encoder layers.
- `decoders.py` — All decoder heads that convert transformer output into child model parameters. Contains `LinearModelDecoder`, `AdditiveModelDecoder`, `FactorizedAdditiveModelDecoder`, `SummaryLayer` (dataset-level summary via attention/averaging), `MLPModelDecoder` (predicts MLP weights/biases), `GradTreeDecoder` (predicts tree I_logits/T/L params), and `GrandeDecoder` (predicts GRANDE split_values/split_index_logits/estimator_weights/leaf_classes with feature statistics conditioning).
- `encoders.py` — Input encoders: `Linear`, `NanHandlingEncoder`, `BinEmbeddingEncoder`, `OneHotAndLinear`, and `get_fourier_features`.
- `gamformer.py` — GAMformer model: bi-attention transformer producing additive (per-feature shape function) predictions.
- `grande_core.py` — Shared GRANDE utilities: `build_grande_context` (estimator feature subset sampling), `build_grande_feature_stats` (per-estimator feature statistics), `grande_forward` (differentiable tree ensemble forward pass with ST estimators and missing-value routing), `build_tree_index_tensors`, `one_hot_argmax`, `st`, and `resolve_selected_variables`.
- `la_mothernet.py` — `SSMMotherNet`: MotherNet variant using linear attention (or other SSM) backbone instead of standard transformer.
- `layer.py` — Core transformer building blocks: `TransformerEncoderLayer`, `TransformerEncoderSimple`, `BiAttentionEncoderLayer`, and `LinearBiAttentionEncoderLayer`.
- `linear_attention.py` — Linear attention implementation: `LinearAttention`, `AttentionLayer`, feature maps (elu, hedgehog, identity), `LinearAttentionTransformerEncoderLayer`, and `get_linear_attention_layers` factory.
- `mothernet.py` — Main MotherNet model. `ModelPredictor` base class handles forward dispatch for mlp/gradtree/grande child models (including `tree_forward` for GradTree). `MotherNet` subclass wires up the standard transformer encoder backbone, input encoder, and chosen decoder.
- `mothernet_additive.py` — `MotherNetAdditive`: MotherNet variant for additive models using bin-encoded inputs and shape-function decoders.
- `perceiver.py` — `TabPerceiver`: Perceiver-based architecture with cross-attention from latents to input data, used as a MotherNet backbone.
- `positional_encodings.py` — Positional encoding variants: `NoPositionalEncoding`, sinusoidal `PositionalEncoding`, `LearnedPositionalEncoding`, and `PairedScrambledPositionalEncodings`.
- `tabflex.py` — `TabFlex`: lightweight tabular model using linear attention backbone with a simple MLP decoder head.
- `tabpfn.py` — `TabPFN`: standard transformer-based prior-fitted network for tabular classification.
- `utils.py` — Utility functions: `bin_data` (quantile-based feature binning to one-hot) and `sklearn_like_binning` helper.
