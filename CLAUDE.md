# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a **MotherNet for GRANDE** — instead of having MotherNet's transformer backbone predict MLP weights, have it predict the parameters of a GRANDE-style differentiable decision tree ensemble. The relevant files are:

- `grande.py` — standalone GRANDE implementation (reference for the tree algorithm)
- `ticl/models/mothernet.py` — MotherNet model with `ModelPredictor` (forward/tree_forward) and `MotherNet` (architecture)
- `ticl/models/decoders.py` — `GradTreeDecoder`, `GrandeDecoder`, and `MLPModelDecoder`
- `ticl/models/grande_core.py` — shared GRANDE context sampling, feature statistics, and tree forward kernel

## Architecture Overview

### MotherNet Pipeline (ticl/models/mothernet.py)

```
Training data (x, y) → Encoder → TransformerEncoder → Decoder → Child model params → Inference on x_test
```

1. **Encoder**: `Linear` maps input features to `emsize`-dim embeddings. Y is encoded and added to X embeddings.
2. **Transformer backbone**: Standard `TransformerEncoderSimple` (batch_first=False). Produces per-sample contextual embeddings.
3. **Decoder**: Conditioned on transformer output + training labels, produces child model parameters.
4. **Child model forward**: Uses predicted params to run inference on test data within `ModelPredictor.forward()`.

### Child Model Dispatch (`child_model` param)

- `"mlp"` → `MLPModelDecoder` → predicts (bias, weight) tuples for each MLP layer → matrix multiply chain in `forward()`
- `"gradtree"` → `GradTreeDecoder` → predicts (I_logits, T, L) → differentiable tree pass in `tree_forward()`
- `"grande"` → `GrandeDecoder` → predicts GRANDE-style `(split_values, split_index_logits, estimator_weights, leaf_classes)` and uses the shared `grande_forward()` kernel

### GradTreeDecoder (ticl/models/decoders.py:561)

Predicts flat parameter vector, then reshapes into per-estimator tree params:
- **I_logits**: `(batch, n_estimators, n_nodes, n_features)` — feature selection logits per internal node
- **T**: `(batch, n_estimators, n_nodes)` — scalar split threshold per node (no feature dim — unnecessary when predicted by a hypernetwork)
- **L**: `(batch, n_estimators, n_leaves, n_out)` — leaf class logits

Total params per tree: `n_nodes * in_size + n_nodes + n_leaves * n_out`. Multiplied by `n_estimators`.

Precomputes `path_identifier_list` and `internal_node_index_list` (registered as non-grad parameters) for mapping leaf→internal-node paths during inference.

### tree_forward (ticl/models/mothernet.py:35)

Implements Algorithm 1 from the GradTree/GRANDE paper with straight-through (ST) estimators:
1. Feature selection: softmax + ST hard one-hot on I_logits (masks padding features with -inf)
2. Split probability: `softsign((threshold - selected_feature_value)) + 1) / 2`, then ST round
3. Path probability: product over depth using precomputed path indices
4. Leaf aggregation: einsum over path probs and leaf logits
5. Ensemble average over estimators

### GRANDE_Module (grande.py:165) — Reference

Standalone GRANDE with its own learnable parameters (split_values, split_index_array, estimator_weights, leaf_classes_array). Key differences from the MotherNet GradTree version:
- Has per-estimator feature subset selection (`features_by_estimator`)
- Uses per-estimator learned weights for weighted aggregation (softmax over `estimator_weights`)
- Supports data subsetting per estimator (`data_subset_fraction`, `bootstrap`)
- Has dropout on estimator weights
- Uses separate learning rates for splits, indices, weights, and leaves
- Missing value handling via nan masking

### SummaryLayer (ticl/models/decoders.py:285)

Shared between MLPModelDecoder and GradTreeDecoder. Converts per-sample transformer output into a fixed-size dataset-level summary. Supports multiple `decoder_type` strategies: `output_attention` (default, uses cross-attention with learned query), `special_token`, `class_tokens`, `class_average`, `average`.

## Commands

```bash
# Install
conda create -f environment.yml && conda activate ticl && pip install -e .

# Train MotherNet (MLP child)
python ticl/fit_model.py mothernet

# Train MotherNet (GradTree child)
python ticl/fit_model.py mothernet --child-model gradtree --tree-depth 5 --n-estimators 10

# Specify GPU
python ticl/fit_model.py mothernet -g 0

# See all options
python ticl/fit_model.py mothernet -h
```

## Key Config (ticl/model_configs.py)

MotherNet defaults include `child_model: "mlp"`, `tree_depth: 5`, `n_estimators: 1`. Override via CLI args (`--child-model`, `--tree-depth`, `--n-estimators`). Config is parsed in `ticl/cli_parsing.py`.

## Tensor Conventions

- Transformer operates **batch_first=False**: sequences are `(seq_len, batch, emsize)`
- `x` input: `(n_samples, batch, n_features)`, split at `single_eval_pos` into train/test
- `info` dict may contain `num_features_used` for masking padded features in tree inference
- Decoders output per-batch parameters; test inference is vectorized over `n_test` samples

## GradTree Code Paths — Keep In Sync

The GradTree child model has **three code paths** that must stay consistent when changing tensor shapes or tree logic:

1. **Training forward**: `ticl/models/mothernet.py` → `tree_forward()` (differentiable, uses ST operators)
2. **Parameter extraction**: `ticl/prediction/mothernet.py` → `extract_gradtree_model()` (runs decoder, squeezes batch dim)
3. **Standalone inference**: `ticl/prediction/mothernet.py` → `predict_with_gradtree_model()` (numpy CPU path + torch CUDA path)

Any change to decoder output shapes (in `ticl/models/decoders.py` → `GradTreeDecoder`) must be propagated to all three.

## Current Simplifications Vs `grande.py`

These notes apply to the legacy `child_model="gradtree"` path. The new `child_model="grande"` path is the closer GRANDE integration and uses `ticl/models/grande_core.py`.

- **Estimator aggregation**: MotherNet averages estimator logits uniformly. `grande.py` uses instance-dependent softmax weights derived from `estimator_weights` and the active leaf per estimator, with optional dropout.
- **Missing values**: MotherNet currently zero-imputes NaNs before tree evaluation. `grande.py` has nan-aware routing that sends masked splits down both branches with learned probabilities.
- **Feature subsets per estimator**: MotherNet predicts split logits over all padded features for every estimator. `grande.py` samples `features_by_estimator` and each estimator only chooses among its subset.
- **Estimator data subsampling**: MotherNet does not model `data_subset_fraction` / `bootstrap`.
- **Parameterization**: MotherNet predicts scalar threshold `T` per node directly. `grande.py` learns `split_values` and `split_index_array` separately, then combines them into node thresholds.

When comparing outputs against `grande.py`, do not assume parity unless these gaps are intentionally closed.

## Review Checklist For Future GradTree Changes

- If you add estimator weights, update both `tree_forward()` and `predict_with_gradtree_model()`; the current ensemble reduction is a plain mean.
- If you add nan-aware routing, do it in both training and standalone inference. Right now both CPU and CUDA inference paths zero-fill NaNs.
- If you change feature selection or threshold semantics, verify all three GradTree code paths plus `extract_gradtree_model()`.
- `GradTreeDecoder` stores `path_identifier_list` and `internal_node_index_list` as non-trainable parameters in the state dict. Any migration of those tensors affects checkpoint compatibility.
- `MotherNetClassifier.fit()` accepts `model_type == "la_mothernet"`, but the extraction helpers in `ticl/prediction/mothernet.py` only know how to run `transformer_encoder`, `linear_attention`, or perceiver-style models. If GradTree is used with `SSMMotherNet`, those helpers must be extended to call `model.ssm` / `model.inner_forward()`.

## GRANDE Path Notes

- `child_model="grande"` is the end-to-end GRANDE-style path. It uses a fixed estimator-local feature budget (`selected_variables`) instead of predicting over the full padded feature axis.
- `GrandeDecoder` does **not** predict `features_by_estimator`. Those feature subsets are sampled externally via `build_grande_context()` and must be carried through extraction/inference.
- The GRANDE path uses one shared torch kernel (`grande_forward`) for training forward and extracted inference. Avoid reintroducing separate numpy/CUDA implementations unless there is a strong reason.
- Row subsampling / bootstrap affect the estimator-local feature statistics used by `GrandeDecoder`, not the inference-time tree kernel directly.
