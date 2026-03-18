# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a **MotherNet for GRANDE**: instead of having MotherNet's transformer backbone predict MLP weights, have it predict the parameters of a GRANDE-style differentiable decision tree ensemble.

The most relevant files are:

- `grande.py` — standalone GRANDE implementation (reference for the tree algorithm)
- `ticl/models/grande_core.py` — shared GRANDE context sampling, feature statistics, and tree forward kernel
- `ticl/models/decoders.py` — `GradTreeDecoder`, `GrandeDecoder`, and `MLPModelDecoder`
- `ticl/models/mothernet.py` — MotherNet model with `ModelPredictor.forward()` and `MotherNet`
- `ticl/prediction/mothernet.py` — GRANDE extraction and standalone inference helpers
- `ticl/tests/test_grande_core.py` — focused tests for GRANDE context, stats, decoder shapes, and kernel behavior

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

### GrandeDecoder (ticl/models/decoders.py:719)

`GrandeDecoder` is the current MotherNet-side GRANDE implementation. It does not learn tree parameters directly as model parameters; it predicts them from the dataset representation produced by MotherNet.

Per estimator, the decoder consumes:
- A dataset-level summary from `SummaryLayer`
- Estimator-local feature statistics from `build_grande_feature_stats()`
- A learned estimator embedding

It outputs:
- `split_values`: `(batch, n_estimators, n_nodes, selected_variables)`
- `split_index_logits`: `(batch, n_estimators, n_nodes, selected_variables)`
- `estimator_weights`: `(batch, n_estimators, n_leaves)`
- `leaf_classes`: `(batch, n_estimators, n_leaves, n_out)`

Current decoder behavior that matters:
- `features_by_estimator` and `feature_mask` come from external context built by `build_grande_context()`
- Feature stats include per-feature mean, std, missing rate, and class-conditional means
- The final decoder layer is initialized with small random noise for symmetry breaking across estimators
- Raw predicted `split_values` are transformed from z-space into feature space using estimator-local mean and std
- `leaf_classes` receive a dataset class-prior logit residual
- When `return_profile=True`, the decoder returns timing data for feature-stat construction and the decoder MLP

### grande_forward (ticl/models/grande_core.py:268)

`grande_forward()` is the shared tree kernel used by both training and extracted inference.

It implements a hard, axis-aligned, differentiable ensemble:
1. Gather estimator-local features using `features_by_estimator`
2. Apply masked softmax over `split_index_logits`, optionally temperature-scaled, then take a hard argmax
3. Use a straight-through estimator during training to keep gradients through hard feature selection
4. Compute node decisions with `softsign(split_value - selected_feature_value)` and hard-round them
5. Handle missing values by routing masked splits down both branches with learned probabilities
6. Accumulate leaf path probabilities in log space for numerical stability
7. Use path-conditioned `estimator_weights` to softmax-weight estimators per sample
8. Aggregate weighted leaf logits into final class logits

### GRANDE_Module (grande.py:165) — Reference

Standalone GRANDE with its own learnable parameters (`split_values`, `split_index_array`, `estimator_weights`, `leaf_classes_array`).

Key reference properties:
- Has per-estimator feature subset selection (`features_by_estimator`)
- Uses per-estimator learned weights for weighted aggregation (softmax over `estimator_weights`)
- Supports data subsetting per estimator (`data_subset_fraction`, `bootstrap`)
- Has dropout on estimator weights
- Uses separate learning rates for splits, indices, weights, and leaves
- Missing value handling via nan masking

### SummaryLayer (ticl/models/decoders.py:285)

Shared between `MLPModelDecoder`, `GradTreeDecoder`, and `GrandeDecoder`. Converts per-sample transformer output into a fixed-size dataset-level summary. Supports multiple `decoder_type` strategies: `output_attention` (default, uses cross-attention with learned query), `special_token`, `class_tokens`, `class_average`, `average`.

## Commands

```bash
# Install
conda create -f environment.yml && conda activate ticl && pip install -e .

# Train MotherNet (MLP child)
python ticl/fit_model.py mothernet

# Train MotherNet (GradTree child)
python ticl/fit_model.py mothernet --child-model gradtree --tree-depth 5 --n-estimators 10

# Train MotherNet (GRANDE child)
python ticl/fit_model.py mothernet --child-model grande --tree-depth 4 --n-estimators 64

# Specify GPU
python ticl/fit_model.py mothernet -g 0

# See all options
python ticl/fit_model.py mothernet -h
```

## Key Config (ticl/model_configs.py)

MotherNet defaults include `child_model: "mlp"`, `tree_depth: 5`, `n_estimators: 1`. Override via CLI args. For the GRANDE path, the main config knobs are:

- `--child-model grande`
- `--tree-depth`
- `--n-estimators`
- `--selected-variables`
- `--data-subset-fraction`
- `--bootstrap`
- `--grande-dropout`
- `--missing-values`
- `--grande-random-state`
- `--split-temperature`
- `--grande-profile`

Config is parsed in `ticl/cli_parsing.py` and defaulted in `ticl/model_configs.py`.

## Tensor Conventions

- Transformer operates **batch_first=False**: sequences are `(seq_len, batch, emsize)`
- `x` input: `(n_samples, batch, n_features)`, split at `single_eval_pos` into train/test
- `info` dict may contain `num_features_used` for masking padded features in tree inference
- Decoders output per-batch parameters; test inference is vectorized over `n_test` samples
- Extracted GRANDE models squeeze the training batch dimension and store `(n_estimators, ...)` tensors for standalone inference

## GradTree Code Paths — Keep In Sync

The GradTree child model has **three code paths** that must stay consistent when changing tensor shapes or tree logic:

1. **Training forward**: `ticl/models/mothernet.py` → `tree_forward()` (differentiable, uses ST operators)
2. **Parameter extraction**: `ticl/prediction/mothernet.py` → `extract_gradtree_model()` (runs decoder, squeezes batch dim)
3. **Standalone inference**: `ticl/prediction/mothernet.py` → `predict_with_gradtree_model()` (numpy CPU path + torch CUDA path)

Any change to decoder output shapes (in `ticl/models/decoders.py` → `GradTreeDecoder`) must be propagated to all three.

## GRANDE Code Paths — Keep In Sync

The GRANDE child model also has **three code paths**:

1. **Training forward**: `ticl/models/mothernet.py` → grande branch in `ModelPredictor.forward()`
2. **Parameter extraction**: `ticl/prediction/mothernet.py` → `extract_grande_model()`
3. **Standalone inference**: `ticl/prediction/mothernet.py` → `predict_with_grande_model()`

Important details:
- `grande_forward()` is shared across training and standalone inference, so kernel changes usually propagate automatically
- Changes to decoder output names, shapes, normalization, or metadata still must be reflected in extraction and standalone inference
- `features_by_estimator`, `feature_mask`, `path_identifier_list`, `internal_node_index_list`, `missing_values`, `split_temperature`, and `feature_rescale` are part of the extracted model contract

## Legacy GradTree Vs GRANDE

These notes apply to the legacy `child_model="gradtree"` path. The newer `child_model="grande"` path is the actual GRANDE-style integration and uses `ticl/models/grande_core.py`.

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

- `child_model="grande"` is the end-to-end GRANDE-style path that produces a hard, axis-aligned differentiable tree ensemble.
- `selected_variables` is a fixed estimator-local feature budget. If the configured value is `<= 1`, it is interpreted as a fraction of max features and clamped into the GRANDE-style min/max range in `resolve_selected_variables()`.
- `GrandeDecoder` does **not** predict `features_by_estimator`; they are sampled externally via `build_grande_context()` and must be carried through extraction and inference.
- Training currently builds GRANDE context in the forward pass without an explicit seed. Extraction uses `grande_random_state` to freeze one deterministic context draw.
- `build_grande_feature_stats()` computes estimator-local feature summaries, and optional row subsampling or bootstrap only affects those summaries.
- `predict_with_grande_model()` preserves NaNs so the shared kernel can use `missing_values=True` routing at inference time.
- Extracted inference multiplies normalized features by `feature_rescale = max_features / X_train.shape[1]` to match the padded-feature representation seen during extraction.
- Runtime hot spots are in `ticl/models/grande_core.py`, especially context sampling and estimator-local feature statistics. Keep those paths vectorized.
- Use `--grande-profile True` when investigating runtime. It records epoch-level timings for `grande_context_s`, `grande_feature_stats_s`, `grande_decoder_mlp_s`, and `grande_forward_s`.

## Concerns And Deviations From Original GRANDE

- **Deviation**: The MotherNet GRANDE path is a hypernetwork-predicted tree ensemble. Unlike `grande.py`, the tree parameters are not direct `nn.Parameter`s trained with separate optimizer groups and learning rates.
- **Deviation**: `features_by_estimator` are not persistent model parameters in the MotherNet path. They are sampled outside the decoder via `build_grande_context()`.
- **Deviation**: During training, estimator feature subsets are currently resampled per forward unless a seed is explicitly passed. During extraction, `grande_random_state` freezes a single deterministic draw.
- **Deviation**: Decoder `split_values` are not used as raw learned thresholds. They are predicted in z-space and then mapped into estimator-local feature space using mean and std.
- **Deviation**: Decoder `leaf_classes` receive a dataset class-prior residual before tree aggregation.
- **Deviation**: Path probabilities are accumulated in log space for stability instead of plain `torch.prod`.
- **Deviation**: `data_subset_fraction` and `bootstrap` affect the estimator-local feature statistics passed into the decoder, not the inference-time tree kernel structure itself.
- **Concern**: `build_grande_context()` is performance-sensitive and device-sensitive. If you change it, verify CPU and CUDA assignment semantics carefully.
- **Concern**: Extracted GRANDE inference still inherits MotherNet's prior/config limits, including the padded feature cap from `config["prior"]["num_features"]`.
- **Concern**: Benchmark parity with reference GRANDE or with CART should not be assumed from architecture alone; benchmark notebooks currently include mixed results and some dataset failures.

When comparing against `grande.py`, treat the current `child_model="grande"` path as a close GRANDE-style implementation, not an exact reproduction.
