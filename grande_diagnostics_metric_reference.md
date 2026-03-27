# GRANDE Diagnostics Metric Reference

This document describes the metrics logged under `grande_diagnostics/*` for GRANDE MotherNet runs.

Scope:
- Metrics are logged at initialization (`epoch=0`) and after every training epoch.
- Most metrics are forward-only and always available when `grande_diagnostics=True`.
- Gradient metrics are only present when `grande_diagnostics_gradients=True`.
- Histogram keys are stored in W&B as bucketed histograms, not raw tensors.

The implementation lives in `ticl/grande_diagnostics.py`.

## Availability

| Prefix | When present | Notes |
| --- | --- | --- |
| `grande_diagnostics/meta/*` | always | bookkeeping and variant state |
| `grande_diagnostics/output/*` | always | fixed diagnostic snapshot, not validation metrics |
| `grande_diagnostics/feature_selection/*` | always | depth-wise split-feature behavior |
| `grande_diagnostics/thresholds/*` | always | selected-feature thresholds only |
| `grande_diagnostics/diversity/*` | always | tree similarity and effective ensemble dimension |
| `grande_diagnostics/estimator_weights/*` | always | ensemble-weight collapse vs spread |
| `grande_diagnostics/routing/*` | always | how probability mass moves through leaves |
| `grande_diagnostics/context_seed/*` | always | seed sensitivity of the cached diagnostic batch |
| `grande_diagnostics/hists/*` | depends on `grande_diagnostics_level` | absent in `scalars` mode |
| `grande_diagnostics/gradients/*` | only when `grande_diagnostics_gradients=True` | separate diagnostic backward pass |

## Meta

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/meta/epoch` | Epoch index for the diagnostic snapshot. | `0` is the pre-training initialization snapshot. |
| `grande_diagnostics/meta/gradient_metrics_enabled` | Whether the current diagnostic pass also ran the optional backward pass. | `0` means forward-only diagnostics. |
| `grande_diagnostics/meta/split_temperature` | Current split-logit temperature used by the decoder. | Most meaningful for `depthwise_factorized_stats`. For non-depthwise variants this is currently fixed at `1.0`. |

## Output Quality

These are computed on the fixed cached diagnostic batch, not the evaluation set.

### Present for all classification runs

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/output/loss` | Task loss on the diagnostic snapshot. | Direct measure of fit on the fixed batch. |
| `grande_diagnostics/output/nan_share` | Fraction of per-example losses that were NaN before reduction. | Should stay at `0`. |
| `grande_diagnostics/output/accuracy` | Classification accuracy on the diagnostic snapshot. | Higher is better. |
| `grande_diagnostics/output/brier` | Brier score of the predictive probabilities. | Lower is better; penalizes miscalibration. |
| `grande_diagnostics/output/ece` | Expected calibration error computed from confidence bins. | Lower is better. |
| `grande_diagnostics/output/confidence_mean` | Mean predicted confidence. | Useful with `accuracy` and `ece` to detect overconfidence. |
| `grande_diagnostics/output/margin_mean` | Mean class-probability margin. | Higher usually means sharper decisions. |
| `grande_diagnostics/output/margin_std` | Standard deviation of prediction margins. | Very low values can mean uniformly flat or uniformly sharp predictions. |

### Binary-only

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/output/positive_prediction_rate` | Fraction of positive predictions. | Useful for class-collapse or threshold-bias detection. |
| `grande_diagnostics/output/f1` | Binary F1 score on the diagnostic snapshot. | Higher is better. |
| `grande_diagnostics/output/roc_auc` | Binary ROC-AUC on the diagnostic snapshot. | Higher is better. |

### Multiclass-only

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/output/f1_macro` | Macro-averaged F1 across classes. | Higher is better; sensitive to minority-class failure. |
| `grande_diagnostics/output/roc_auc_ovr_macro` | Macro one-vs-rest ROC-AUC across classes. | Higher is better. |

## Feature Selection by Depth

These metrics are logged for every tree depth as:
- `grande_diagnostics/feature_selection/depth_{d}/...`

Here `depth_0` is the root split, `depth_1` the next layer, and so on.

| Suffix | Meaning | How to read it |
| --- | --- | --- |
| `entropy` | Entropy of the split-feature probability distribution at that depth. | Lower means more peaked feature choice; very low can indicate collapse. |
| `mean_max_prob` | Mean top-1 split-feature probability. | Higher means the decoder is more certain about one feature. |
| `top1_top2_margin` | Mean gap between the largest and second-largest split-feature probabilities. | Higher means clearer winner-take-all selection. |
| `unique_ratio` | Ratio of unique selected features to total selected positions at that depth. | Lower means many estimators or nodes reuse the same features. |
| `concentration_hhi` | Herfindahl-Hirschman concentration of selected-feature frequencies. | Higher means stronger feature concentration and likely collapse. |

Interpretation:
- Low `entropy`, low `unique_ratio`, and high `concentration_hhi` together are the strongest sign of feature-selection collapse.
- A single peaked split at one node is not automatically bad. The real concern is systematic reuse across estimators, depths, or batches.

## Threshold Diagnostics by Depth

These metrics are logged for every tree depth as:
- `grande_diagnostics/thresholds/depth_{d}/...`

Important: these are computed only for the feature that was actually selected at each node.

| Suffix | Meaning | How to read it |
| --- | --- | --- |
| `mean` | Mean selected threshold value at that depth. | Useful mainly as a raw-location sanity check. |
| `std` | Standard deviation of selected thresholds at that depth. | Very low values can indicate threshold collapse. |
| `max_abs` | Largest absolute selected threshold value. | Large values can flag unstable or extreme thresholds. |
| `zscore_abs_mean` | Mean absolute z-score of thresholds relative to the selected feature mean and std. | High values mean splits are far from the feature bulk. |
| `variance` | Variance of selected thresholds at that depth. | Another collapse-vs-spread measure. |
| `extreme_fraction_abs_z_gt_3` | Fraction of selected thresholds whose absolute z-score exceeds `3`. | High values mean many unusually extreme thresholds. |

Interpretation:
- Low `std` and low `variance` across training can indicate threshold collapse.
- High `zscore_abs_mean` or high `extreme_fraction_abs_z_gt_3` can indicate unstable or unrealistic splits.

## Diversity

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/diversity/split_index_cosine_mean` | Mean pairwise cosine similarity of estimator split-index logits. | Higher means estimators choose features in more similar ways. |
| `grande_diagnostics/diversity/threshold_cosine_mean` | Mean pairwise cosine similarity of selected-threshold vectors. | Higher means estimators place splits more similarly. |
| `grande_diagnostics/diversity/leaf_cosine_mean` | Mean pairwise cosine similarity of leaf output tensors. | Higher means estimator predictions are more similar. |
| `grande_diagnostics/diversity/effective_dim_95` | Number of singular directions needed to explain 95% of tree-vector variance. | Lower means the ensemble lives in a smaller subspace and is less diverse. |

Interpretation:
- High cosine similarity and low `effective_dim_95` mean the ensemble is collapsing toward redundant trees.

## Estimator Weights

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/estimator_weights/entropy_mean` | Entropy of the estimator-weight distribution. | Higher means more uniform weighting across estimators. |
| `grande_diagnostics/estimator_weights/effective_size_mean` | Effective number of estimators, computed as `1 / sum(w^2)`. | Lower means only a few estimators dominate. |
| `grande_diagnostics/estimator_weights/max_weight_mean` | Mean maximum estimator weight. | Higher means stronger concentration onto one estimator. |

Interpretation:
- These metrics matter most together with the diversity metrics. Uniform weights are less concerning if the ensemble itself is diverse.

## Routing

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/routing/path_entropy_mean` | Entropy of the leaf-path distribution. | Lower means routing is more deterministic. |
| `grande_diagnostics/routing/leaf_occupancy_mean` | Mean occupancy of leaves under the path distribution. | Low occupancy can indicate underused parts of the tree. |
| `grande_diagnostics/routing/dead_leaf_fraction` | Fraction of leaves whose average occupancy is below `1e-4`. | Higher means more dead or unreachable leaves. |
| `grande_diagnostics/routing/missing_value_routing_fraction` | Fraction of selected split positions that route on missing values. | Useful for checking whether NaN-aware routing is materially active. |

Interpretation:
- High `dead_leaf_fraction` is a direct sign that parts of the tree are not participating.
- Very low `path_entropy_mean` with high dead-leaf fraction can mean the ensemble is routing almost all mass through a small subset of leaves.

## Context-Seed Stability

The diagnostic batch is rerun across a small panel of context seeds.

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/context_seed/loss_mean` | Mean loss across the seed panel. | Baseline level of fit under seed perturbation. |
| `grande_diagnostics/context_seed/loss_std` | Standard deviation of loss across the seed panel. | Higher means stronger seed sensitivity. |
| `grande_diagnostics/context_seed/loss_range` | Max-min loss gap across the seed panel. | Another seed-instability measure. |
| `grande_diagnostics/context_seed/probability_variance_mean` | Mean variance of predicted probabilities across seeds. | Higher means unstable predictions. |
| `grande_diagnostics/context_seed/prediction_agreement_mean` | Mean pairwise agreement of hard predictions across seeds. | Lower means stronger seed sensitivity. |

Interpretation:
- A model can have reasonable average loss while still being highly unstable under context reseeding.

## Histograms

These are only logged when `grande_diagnostics_level` is not `scalars`.

| Histogram key | Meaning | Notes |
| --- | --- | --- |
| `grande_diagnostics/hists/selected_feature_depth_{d}` | Distribution of actually selected feature ids at depth `d`. | Best used to visualize feature collapse over time. |
| `grande_diagnostics/hists/threshold_zscore_depth_{d}` | Distribution of selected-threshold z-scores at depth `d`. | Shows whether thresholds are central, broad, or extreme. |
| `grande_diagnostics/hists/effective_estimator_size` | Distribution of effective ensemble size across diagnostic cases. | Useful for spotting per-batch estimator dominance. |
| `grande_diagnostics/hists/leaf_occupancy` | Distribution of leaf occupancies. | Shows whether many leaves are dead or only lightly used. |
| `grande_diagnostics/hists/backbone_activation_grad_norms` | Distribution of backbone activation gradient norms. | Gradient mode only. |
| `grande_diagnostics/hists/backbone_param_grad_norms` | Distribution of backbone parameter-group gradient norms. | Gradient mode only. |

## Gradient Metrics

These are only present when `grande_diagnostics_gradients=True`.

### Parameter gradient groups

For every parameter group below, the following suffixes are logged:
- `/l2_norm`
- `/mean_abs`
- `/max_abs`
- `/grad_to_weight_ratio`
- `/num_params`

Groups:
- `grande_diagnostics/gradients/params/backbone`
- `grande_diagnostics/gradients/params/encoder`
- `grande_diagnostics/gradients/params/y_encoder`
- `grande_diagnostics/gradients/params/transformer_encoder`
- `grande_diagnostics/gradients/params/transformer_layer_{i}`
- `grande_diagnostics/gradients/params/decoder`
- `grande_diagnostics/gradients/params/decoder_{subgroup}`

Interpretation:
- Very small norms across deeper layers can indicate vanishing gradients.
- Very large `grad_to_weight_ratio` can indicate unstable updates.
- Comparing `backbone` against `decoder` tells you whether learning is concentrated only in the head.

### Activation gradient groups

For every activation below, the following suffixes are logged:
- `/l2_norm`
- `/mean_abs`
- `/max_abs`

Activations:
- `grande_diagnostics/gradients/activations/encoder_output`
- `grande_diagnostics/gradients/activations/transformer_layer_{i}`
- `grande_diagnostics/gradients/activations/transformer_output`
- `grande_diagnostics/gradients/activations/split_values`
- `grande_diagnostics/gradients/activations/split_index_logits`
- `grande_diagnostics/gradients/activations/estimator_weights`
- `grande_diagnostics/gradients/activations/leaf_classes`

Interpretation:
- These tell you where signal is dying or exploding in the forward path, rather than only at parameter tensors.

### Per-depth split gradients

| Metric pattern | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/gradients/per_depth/split_values_depth_{d}_l2_norm` | Gradient norm on split values at depth `d`. | Low deep-layer values suggest weak threshold learning deeper in the tree. |
| `grande_diagnostics/gradients/per_depth/split_index_logits_depth_{d}_l2_norm` | Gradient norm on split-index logits at depth `d`. | Low deep-layer values suggest weak feature-selection learning deeper in the tree. |
| `grande_diagnostics/gradients/per_depth/split_values_depth_{d}_cosine_mean` | Mean pairwise cosine similarity of per-estimator split-value gradients at depth `d`. | Higher means estimators are receiving more similar threshold-learning signal at that depth. |
| `grande_diagnostics/gradients/per_depth/split_index_logits_depth_{d}_cosine_mean` | Mean pairwise cosine similarity of per-estimator split-index-logit gradients at depth `d`. | Higher means estimators are receiving more similar feature-selection signal at that depth. |

### Per-estimator gradient similarity

These summarize whether different estimators are being pushed in similar directions during the diagnostic backward pass.

| Metric | Meaning | How to read it |
| --- | --- | --- |
| `grande_diagnostics/gradients/per_estimator/split_index_logits_cosine_mean` | Mean pairwise cosine similarity of per-estimator split-index-logit gradients. | Higher means more shared feature-selection supervision across estimators. |
| `grande_diagnostics/gradients/per_estimator/split_values_cosine_mean` | Mean pairwise cosine similarity of per-estimator split-value gradients. | Higher means more shared threshold-learning supervision across estimators. |
| `grande_diagnostics/gradients/per_estimator/estimator_weights_cosine_mean` | Mean pairwise cosine similarity of per-estimator estimator-weight gradients. | Higher means estimator weighting is being updated more uniformly across trees. |
| `grande_diagnostics/gradients/per_estimator/leaf_classes_cosine_mean` | Mean pairwise cosine similarity of per-estimator leaf-output gradients. | Higher means leaves are receiving more homogeneous prediction signal. |
| `grande_diagnostics/gradients/per_estimator/combined_tree_output_cosine_mean` | Mean pairwise cosine similarity after concatenating all decoder-output gradients per estimator. | Highest-level check for whether the whole tree output is receiving near-copy supervision. |
| `grande_diagnostics/gradients/per_estimator/{name}_effective_dim_95` | Effective estimator-dimension needed to explain 95% of gradient variance for gradient group `{name}`. | Lower means gradient supervision lives in a smaller estimator subspace. |
| `grande_diagnostics/gradients/per_estimator/{name}_effective_dim_fraction` | `effective_dim_95 / n_estimators` for gradient group `{name}`. | Lower means stronger low-rank or shared-gradient structure. |

Here `{name}` is one of:
- `split_index_logits`
- `split_values`
- `estimator_weights`
- `leaf_classes`
- `combined_tree_output`

## Practical Reading Guide

If you are debugging collapse, the most informative starting subset is:
- `feature_selection/depth_{d}/entropy`
- `feature_selection/depth_{d}/unique_ratio`
- `feature_selection/depth_{d}/concentration_hhi`
- `thresholds/depth_{d}/variance`
- `diversity/effective_dim_95`
- `diversity/*_cosine_mean`
- `estimator_weights/effective_size_mean`
- `routing/dead_leaf_fraction`
- `context_seed/loss_std`
- `context_seed/prediction_agreement_mean`

If you are debugging optimization or depth-wise learning failure, add:
- `gradients/params/backbone/*`
- `gradients/params/decoder/*`
- `gradients/per_depth/split_values_depth_{d}_l2_norm`
- `gradients/per_depth/split_index_logits_depth_{d}_l2_norm`
- `gradients/per_estimator/*_cosine_mean`
- `gradients/per_estimator/*_effective_dim_fraction`

## Existing Related Files

- `grande_wandb_diagnostics_plan.md` describes the implementation plan and logging scope.
- `experiments/grande_wandb_analysis.ipynb` is the starter notebook for analyzing these metrics from W&B.
