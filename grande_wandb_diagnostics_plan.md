# GRANDE MotherNet W&B Diagnostics Plan

## Summary

- Add an opt-in GRANDE diagnostics subsystem that logs to W&B at `epoch=0` before any optimizer step, then again after every training epoch.
- Use one deterministic fixed diagnostic snapshot for all structural and gradient comparisons so initialization, epoch-to-epoch drift, and collapse are directly comparable.
- Default logging depth is `scalars + small histograms`, with full backbone and decoder gradient tracking included.

## Interface Changes

- Add new `mothernet` config / CLI flags:
  - `grande_diagnostics: bool = False`
  - `grande_diagnostics_gradients: bool = False`
  - `grande_diagnostics_level: {"scalars","scalars_small_hists","full_hists"} = "scalars_small_hists"`
  - `grande_diagnostics_seed: int = 0`
  - `grande_diagnostics_hist_max_points: int = 2048`
- Keep `grande_profile` unchanged; diagnostics are separate from timing/profile logging.
- Normalize epoch-based W&B logging to use the epoch number as the W&B step. Init diagnostics log at `step=0`.

## Implementation Changes

- Introduce a dedicated diagnostics helper for `child_model="grande"` that can always run a forward-only pass on a cached batch and can optionally run a separate backward pass for gradient metrics without stepping the optimizer.
- At train startup, before the loop:
  - Save Python / NumPy / Torch RNG states.
  - Build one deterministic diagnostic batch from the dataloader using `grande_diagnostics_seed`.
  - Restore RNG states so diagnostics do not perturb normal training randomness.
  - Cache that batch on CPU and log an initialization snapshot at `epoch=0`.
- After each training epoch:
  - Run one diagnostic forward/backward pass on the cached snapshot.
  - Log all diagnostics to W&B, then clear grads so the diagnostic pass does not leak into training.
- Add a private GRANDE debug path that exposes raw tensors needed for metrics:
  - decoder outputs: `split_values`, `split_index_logits`, `estimator_weights`, `leaf_classes`
  - kernel internals: `split_soft`, `node_soft`, `path_probs`, `estimator_weights_softmax`
  - context: `features_by_estimator`, `feature_mask`
- Track these scalar metrics every epoch:
  - Feature selection: per-depth entropy, mean max probability, top1-top2 margin, unique selected features per depth, selected-feature frequency concentration.
  - Thresholds: stats only on the selected feature per node, including mean/std/max abs, z-score vs selected-feature mean/std, extreme-threshold fraction, per-depth variance.
  - Tree diversity: pairwise cosine similarity for `split_index_logits`, selected thresholds, and `leaf_classes`; effective estimator dimension at 95% variance.
  - Estimator weighting: effective ensemble size, weight entropy, mean max estimator weight.
  - Routing: path entropy, leaf occupancy, dead-leaf fraction, missing-value routing fraction.
  - Output quality on the diagnostic batch: loss, accuracy, ROC-AUC/F1 if applicable, class-margin stats, Brier/ECE for classification.
  - Seed/context stability: rerun diagnostics with context seeds `[0, 1, 42]` and log loss/prediction variance plus agreement.
- Track these gradient metrics every epoch from the diagnostic backward pass:
  - Parameter grad summaries for `encoder`, `y_encoder`, full `transformer_encoder`, each transformer layer, full `decoder`, and decoder subgroups by parameter prefix.
  - For each group: grad L2 norm, mean abs grad, max abs grad, grad-to-weight norm ratio.
  - Activation gradient summaries via hooks on encoder output, each transformer block output, transformer final output, and GRANDE tensors (`split_values`, `split_index_logits`, `estimator_weights`, `leaf_classes`).
  - Per-depth grad norms for `split_values` and `split_index_logits` so depth-wise learning failure is visible online.
- Log these bounded histograms each epoch in the default mode:
  - selected feature indices by depth
  - selected-threshold z-scores
  - estimator effective sizes
  - path occupancy / leaf occupancy
  - per-layer backbone grad norms
- Keep diagnostics isolated:
  - run with `optimizer.zero_grad()` before and after
  - no optimizer step, no scheduler step, no checkpoint side effects
  - no change to standard training outputs when diagnostics are disabled

## Follow-Up Note

- Gradient diagnostics are currently split behind `grande_diagnostics_gradients` and default to off.
- Reason: the separate epoch-0 diagnostic backward still triggers a CUDA `invalid argument` failure on the factorized GRANDE smoke run, while all forward-only diagnostics are stable and sufficient for the current collapse/diversity debugging goals.
- Follow-up: isolate the exact GPU backward failure path and re-enable gradient diagnostics in smoke jobs once that root cause is fixed.

## Test Plan

- Unit test that the diagnostic snapshot is deterministic for a fixed diagnostics seed and does not perturb global RNG state.
- Unit test that diagnostics log at `epoch=0` before any optimizer step and once after each epoch.
- Unit test that diagnostic backward populates backbone and GRANDE-head gradient metrics, then leaves model grads cleared afterward.
- Unit test that selected-threshold metrics only use the selected feature, not all candidate features.
- Unit test that W&B payloads contain the expected metric prefixes and bounded histograms in `scalars_small_hists` mode.
- Smoke test one tiny GRANDE training run on CPU with diagnostics enabled and verify no training regression when diagnostics are off.

## Assumptions And Defaults

- Scope is GRANDE MotherNet only; no new diagnostics are added for MLP or legacy GradTree.
- The fixed diagnostic snapshot is the canonical comparison point for init and all epochs.
- Gradient diagnostics are computed from a dedicated diagnostic loss on the cached batch, not from accumulated training gradients.
- Default seed panel for context-stability diagnostics is `[0, 1, 42]`.
- Default W&B mode is `scalars_small_hists`; full parameter histograms are not logged by default to avoid excessive run cost.
