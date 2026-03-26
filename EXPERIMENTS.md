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
