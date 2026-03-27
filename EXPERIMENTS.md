# Experiments Log

Track hypotheses, experiments, and results here. Goal: close the performance gap between GRANDE MotherNet and MLP MotherNet.

## Template

```
### [YYYY-MM-DD] Experiment title
**Hypothesis**: ...
**Test**: ...
**Config**: `python ticl/fit_model.py mothernet --child-model grande ...`
**Result**: ...
**Conclusion**: ...
```

---

## Experiments

### [2026-03-25] Diagnostic: Root causes of GRANDE underperformance
**Hypothesis**: Multiple structural issues prevent GRANDE MotherNet from matching MLP MotherNet: near-random feature selection, seed mismatch, split clustering, low diversity, uniform weighting, miscalibration, gradient vanishing at depth.
**Test**: Load existing checkpoints (factorized epoch 270, depthwise epoch 190) and quantify each hypothesis on breast-w, credit-approval, credit-g datasets. No new training.
**Config**: See `experiments/grande_diagnostics.ipynb`
**Result**: (run notebook to populate)
**Conclusion**: (pending notebook execution)

### [2026-03-26] Diagnostic: W&B run analysis for GRANDE training-time failure modes
**Hypothesis**: The new per-epoch `grande_diagnostics/*` metrics can narrow the GRANDE failure mode to a small set of training-time issues rather than a generic optimization failure.

**Test**: Execute `experiments/grande_wandb_analysis.ipynb` on W&B run `raiizen1-university-of-mannheim/tabflex_new/mn_childmodelgrande_E50_grandedecodervariantfactorized_stats_grandediagnosticsTrue_nestimators64_n2048_treedepth4_0_6e1dcc9d4a65` and inspect init-to-final metric trajectories, per-depth collapse tables, and histogram reconstructions. No new training.

**Config**: `experiments/grande_wandb_analysis.ipynb` against the factorized diagnostics run above (`mothernet_n_estimators=64`, `mothernet_tree_depth=4`, `mothernet_selected_variables=16`; gradient diagnostics disabled for this run).

**Result**: The strongest signal is learned ensemble redundancy. Pairwise estimator similarity rises sharply from init to final: `split_index_cosine_mean 0.302 -> 0.946`, `threshold_cosine_mean 0.051 -> 0.862`, `leaf_cosine_mean 0.301 -> 0.935`. The fixed notebook also shows that the redundancy is substantial but partial, not a near-total collapse: `effective_dim_95 = 46.625` (`effective_dim_fraction = 0.729`) and `effective_size_mean = 44.77` (`effective_size_fraction = 0.700`) at the end of training. Dead routing also worsens during training: `dead_leaf_fraction 0.182 -> 0.295`. `path_entropy_mean` is effectively zero throughout, which is expected under straight-through hard routing, so the meaningful routing signal is dead-leaf growth rather than path-entropy collapse. Threshold collapse is not supported: root threshold `std 0.164 -> 0.982` and `variance 0.027 -> 0.965`, with tiny extreme-threshold fractions. Feature-selection collapse is weaker and is better described as ensemble-level feature reuse than hard per-node collapse: normalized feature entropy remains fairly high even at depth 3 (`2.186 / log(16) = 0.789` of max), while `unique_ratio` falls with depth (`0.496, 0.371, 0.234, 0.130` for depths 0-3). Important caveat: `unique_ratio` here is the ratio of unique selected features to split positions at a given depth, not a literal fraction of the global feature universe. Seed instability is present but does not look dominant at the end of training: `loss_std / loss_mean ~= 0.49%`, `prediction_agreement_mean = 0.881`. Gradient pathologies remain untested because this run did not log gradient diagnostics.

**Conclusion**: For this factorized GRANDE run, the main training-time lead is partial ensemble homogenization, with dead-leaf growth as a coupled downstream effect. Threshold collapse is ruled out on the logged metrics, feature-selection collapse is secondary and mostly reflects deeper feature reuse across estimators rather than low-entropy winner-take-all behavior, and seed instability is not the primary explanation for the final behavior. The next validation step should be to test interventions that explicitly preserve inter-tree diversity and reduce dead routing, then inspect the new `grande_diagnostics_gradients=True` run to distinguish shared-gradient effects from a decoder-architecture bottleneck.

### [2026-03-27] Diagnostic: W&B analysis for factorized run with `data_subset_fraction=0.8`
**Hypothesis**: If reducing estimator-local statistics to `data_subset_fraction=0.8` addresses the right bottleneck, the diagnostics should show less ensemble redundancy and possibly less dead routing than the earlier factorized run. If the same failure modes remain, that would support the hypothesis that the main issue is a shared decoder / objective bottleneck rather than threshold collapse or seed noise.

**Test**: Execute `experiments/grande_wandb_analysis.ipynb` on W&B run `raiizen1-university-of-mannheim/tabflex_new/mn_childmodelgrande_datasubsetfraction0.8_E50_grandedecodervariantfactorized_stats_grandediagnosticsTrue_grandediag_fd5b3759fe66` and compare its scorecard and per-depth tables against the earlier factorized diagnostics run `raiizen1-university-of-mannheim/tabflex_new/mn_childmodelgrande_E50_grandedecodervariantfactorized_stats_grandediagnosticsTrue_nestimators64_n2048_treedepth4_0_6e1dcc9d4a65`. No new training.

**Config**: `experiments/grande_wandb_analysis.ipynb` against the subset run above (`mothernet_data_subset_fraction=0.8`, `mothernet_n_estimators=64`, `mothernet_tree_depth=4`, `mothernet_selected_variables=16`, factorized decoder). The saved notebook output reports final diagnostic `loss = 0.8733` and `accuracy = 0.6018`.

**Result**: The qualitative ordering does not change: the only watch-level modes are still dead routing (`score = 0.613`), ensemble redundancy (`0.612`), and feature-selection collapse (`0.521`). Relative to the earlier factorized run, diversity is slightly worse rather than better: `split_index_cosine_mean 0.946 -> 0.965`, `threshold_cosine_mean 0.862 -> 0.880`, `effective_dim_fraction 0.729 -> 0.643`, and `effective_size_fraction 0.700 -> 0.626` (while `leaf_cosine_mean` changes only slightly, `0.935 -> 0.924`). Dead routing remains elevated but improves slightly rather than disappearing: `dead_leaf_fraction 0.295 -> 0.281`, with `path_entropy_fraction = 0.0` still expected under hard routing. Feature reuse at deeper nodes is somewhat stronger than before: `unique_ratio` becomes `0.584, 0.282, 0.221, 0.125` across depths 0-3 versus `0.496, 0.371, 0.234, 0.130` previously, and depth-3 normalized entropy falls from `0.789 -> 0.632`; however, concentration remains low (`HHI <= 0.047`), so this still looks like ensemble-level reuse rather than hard one-feature collapse. Threshold collapse remains unsupported across all depths (`std` and `variance` stay substantial; extreme threshold fractions are near zero). Seed instability looks even less plausible than before: `loss_cv 0.0049 -> 0.0036`, `prediction_agreement_mean 0.881 -> 0.917`. The saved notebook output also reports little evidence for gradient pathologies (`score = 0.0`, backbone/decoder gradient-L2 ratio `0.526`, deepest/root split-value gradient ratio `0.846`, deepest/root split-index ratio `0.553`).

**Conclusion**: `data_subset_fraction=0.8` does not support the idea that estimator-local row subsetting fixes the core GRANDE failure mode by restoring tree diversity. If anything, it strengthens the existing hypothesis that the dominant problem is learned redundancy from the shared decoder / objective, with dead routing as a coupled but not sole effect. The new output further weakens threshold-collapse, seed-instability, and simple vanishing-gradient explanations. The main new nuance is that row subsetting may slightly increase deeper feature reuse while only marginally reducing dead leaves.

### [2026-03-27] Diagnostic: Per-estimator gradient alignment on factorized checkpoint replay
**Hypothesis**: If the factorized GRANDE MotherNet is collapsing because different estimators receive nearly identical direct supervision, then per-estimator decoder-output gradients should already have high pairwise cosine similarity on a fixed diagnostic backward pass.

**Test**: Load the local factorized checkpoint `models_diff/mn_childmodelgrande_datasubsetfraction0.8_E50_grandedecodervariantfactorized_stats_grandediagnosticsTrue_grandediagnosticsgradientsTrue_nestimators64_n2048_treedepth4_03_26_2026_19_56_18_epoch_50.cpkt`, reconstruct one diagnostic batch from the current dataloader using `prepare_grande_diagnostic_snapshot()`, run one backward pass with `return_debug=True`, and measure pairwise cosine similarity across estimators for retained decoder-output gradients (`split_index_logits`, `split_values`, `estimator_weights`, `leaf_classes`) plus their concatenation. No new training.

**Config**: Ad hoc offline probe using `load_model()`, `get_dataloader()`, `prepare_grande_diagnostic_snapshot()`, and a single backward pass through the checkpoint above on CPU with `grande_context_seed=0`, `advance_split_temperature=False`, and `grande_use_training_schedule=True`.

**Result**: The reconstructed diagnostic batch gave `loss = 1.0959` and `accuracy = 0.4630`, so this was not an exact replay of the notebook snapshot. Even so, per-estimator gradient cosine was low relative to the learned tree redundancy: `split_index_logits_cosine_mean = 0.0125`, `split_values_cosine_mean = 0.0218`, `estimator_weights_cosine_mean = 0.0762`, `leaf_classes_cosine_mean = 0.1065`, and `combined_tree_output_cosine_mean = 0.0624`. The corresponding effective-dimension fractions were `0.398`, `0.281`, `0.131`, `0.375`, and `0.414`. Depth-wise gradient cosine also stayed low for structure and thresholds (`split_index`: `0.0042, 0.0090, 0.0121, 0.0084`; `split_values`: `0.0019, 0.0173, 0.0271, 0.0295` from root to depth 3). These values are far below the final learned redundancy seen in the W&B diagnostics (`split_index_cosine ~0.976`, `threshold_cosine ~0.873`, `leaf_cosine ~0.940` in the gradient-tracked run discussion).

**Conclusion**: On this checkpoint replay, the “different estimators receive almost identical direct gradients” story is not supported. There is some shared low-rank structure in the gradient signal, especially for estimator weights, but not near-copy supervision. The more plausible explanation remains that a shared decoder / symmetry / objective bottleneck maps only moderately aligned supervision into highly redundant trees. Important caveat: because the reconstructed batch did not match the notebook snapshot fit, treat these values as directional evidence rather than a definitive run-matched measurement. This motivated adding in-run per-estimator gradient-cosine diagnostics to `grande_diagnostics.py` so the next diagnostics run can test the hypothesis on the exact logged snapshot.
