# Adversarial Review — GRANDE Implementation

*Generated 2026-04-04 via Codex adversarial review*

Notation: `B=batch_size`, `E=n_estimators`, `V=selected_variables`, `N=2^depth-1`, `L=2^depth`, `O=n_out`, `H=hidden_size`, `S=n_test`.

---

## 1. Design choices under scrutiny

The transformer is collapsed to one dataset summary `x_summary`, then duplicated to `R^{B x E x summary_dim}` and concatenated with `feature_stats.reshape(B,E,V*(3+O))` and estimator embeddings before a decoder predicts all tree parameters. (`decoders.py:337`, `decoders.py:804`, `grande_core.py:317`)

**Concern:** This is a very lossy decomposition. In baseline/factorized variants, an entire tree is decoded from one estimator-level vector; there is no node token or path-local latent. Even in `depthwise_factorized_stats`, only split parameters get recurrent structure, while `estimator_weights` and `leaf_classes` still come from the flat estimator input. **Hypothesis: this is a fundamental bottleneck versus the MLP path, which decodes a dense function directly from the summary.**

**Severity: High.**

---

## 2. Gradient flow and differentiability

`grande_forward` uses ST twice — once for feature choice and once for node routing. (`grande_core.py:340–388`)

**Concern:** The forward pass is hard, but the backward pass pretends it was soft. `s1_sum = einsum(split_values, split_index)` means the direct gradient into `split_values` is multiplied by forward-hard `split_index`, so only the chosen feature channel gets threshold gradient. Likewise `leaf_classes` and `estimator_weights` are trained through `weighted_paths`/`path_probs`, so only active leaves get direct signal. Since `path_probs` is a product over `depth`, this sparsity compounds with tree depth. **Hypothesis: this biased, depth-multiplicative ST graph is a major reason GRANDE lags the dense MLP decoder.**

**Severity: Critical.**

---

## 3. Decoder design

Three decoder variants: joint `baseline`, per-output-head `factorized_stats`, and recurrent-split `depthwise_factorized_stats`. (`decoders.py:815–1047`)

**Concern:** The factorization is not obviously meaningful. All heads read the same estimator vector, so the model loses explicit coupling between split feature, threshold, and leaf value. In the depthwise variant, only split generation depends on recurrent node state; leaves still do not condition on the realized split topology. Also, the variants are confounded: baseline gets explicit `N(0,0.05)` last-layer init, factorized/depthwise do not, and only depthwise uses split-temperature scheduling. Compared with standalone GRANDE, which optimizes these heterogeneous targets with separate learning rates, MotherNet collapses them into shared decoder weights.

**Severity: High.**

---

## 4. Feature selection mechanism

Split logits are softmaxed over `V` candidate features and ST-hardened to one-hot. (`grande_core.py:340–345`)

**Concern:** This creates all-or-nothing competition at each node. A small logit ordering change swaps the entire subtree onto a different feature, but threshold learning for non-argmax channels is effectively dead until they win. Temperature scheduling only affects `depthwise_factorized_stats`; baseline and `factorized_stats` always run at implicit temperature 1.0. **Hypothesis: for MotherNet, this is too brittle an inductive bias relative to dense MLP weight prediction.**

**Severity: High.**

---

## 5. Context and statistics

Per-estimator feature subsets and per-feature statistics are built outside the transformer and fed as conditioning. (`mothernet.py:281`, `grande_core.py:230`, `prediction/mothernet.py:345`)

**Concern:** Training normally re-samples `features_by_estimator` every forward (default `grande_context_seed=None`), but extraction/inference fixes the seed from config. **MotherNet is trained on moving tree coordinates and evaluated on frozen ones.** Separately, `feature_stats` contains only mean, std, missing rate, and class means — no quantiles, ranks, class variances, or pairwise signals. `data_subset_fraction`/`bootstrap` only subsample rows for stats, unlike standalone GRANDE where subsetting changes per-estimator forward data.

**Severity: Critical.**

---

## 6. Assumptions that could be wrong

Missing-value routing uses batch-average branch probability, and estimator aggregation is a softmax competition. (`grande_core.py:358–377`)

**Concern:** When a feature is missing, routing uses `node_soft.mean(dim=0)`, so one sample's prediction depends on other samples in the test batch. Estimators compete through `softmax(estimator_weights_leaf)` with default `grande_dropout=0.0`, even though standalone GRANDE defaults to `dropout=0.2`, and diversity regularization is only available for `factorized_stats`. **Hypothesis: these assumptions encourage estimator collapse and batch-dependent behavior that the MLP path simply does not have.**

**Severity: High.**

---

## 7. Comparison with MLP path

The MLP decoder predicts dense matrices, and every hidden unit receives gradient from every sample-feature interaction through standard matmuls. GRANDE predicts `split_index_logits`, `split_values`, `estimator_weights`, and `leaf_classes`, then routes each sample through hard one-hot feature choice and hard path selection. **The MLP path is a smooth dense regression problem for the decoder; the GRANDE path is a sparse combinatorial control problem on the same summary representation.** That structural mismatch alone can explain a persistent gap.

**Severity: High.**

---

## 8. Scalability and expressiveness

`params_per_tree = 2*N*V + L + L*O`, so tree depth increases decoder output size and leaf sparsity exponentially. As `depth` grows, you add exponentially many leaves but still only supervise one path per sample; as `E` or `V` grows, you increase random-context variance and memory without making nodes any less one-feature-at-a-time.

**Severity: High.**

---

## Top 3 Concrete Experiments

1. **Fix tree coordinates during training.** Change context sampling so `features_by_estimator` is deterministic per task (cache or seed it), eliminating the train/eval mismatch. Measure accuracy and `split_index_logits` stability on repeated forwards of the same task.

2. **Soften routing during warmup.** Use `split_soft`/`node_soft` for the first K steps instead of ST, then anneal to hard. If gradient norms on `split_values`/`leaf_classes` densify and validation improves, the ST graph is the culprit.

3. **Ablate the marginal-stat threshold heuristic.** Replace `mu + delta * sigma` with raw predicted thresholds, or extend feature stats to return quantiles. If that closes the gap, the current conditioning signal is too weak or too restrictive.

---

## Re-evaluation and Prioritization

*Based on manual review of findings against existing experimental evidence (EXPERIMENTS.md).*

### Severity re-ranking

The original review's severities need adjustment based on what the repo's diagnostics actually show:

| Point | Original | Revised | Rationale |
|-------|----------|---------|-----------|
| 5. Context/statistics | Critical | **Critical** | Train/eval coordinate mismatch (`seed=None` during training vs fixed seed at inference) is the strongest new finding. Directly testable and likely silently capping performance. |
| 1. Summary bottleneck | High | **High** | Entire tree decoded from one estimator-level vector with no node-local latent. Structural limitation. |
| 3. Decoder design | High | **High** | Variants are confounded (init, temperature scheduling differ). Factorization is only partial — all heads share the same estimator vector. |
| 4. Feature selection | High | **High** | Hard one-hot with no temperature control in baseline/factorized variants. Brittle inductive bias for MotherNet. |
| 2. ST gradients | Critical → | **Medium-high** | Mechanistically true, but existing diagnostics (EXPERIMENTS.md:45) show little evidence for gradient pathology in factorized runs at current depth. Not yet proven as the primary bottleneck. |
| 6. Assumptions | High | **Medium-high** (aggregation/dropout) / **Medium** (missingness) | Missing-value batch-dependence is real but NaNs are off by default in the prior (`model_configs.py:172`). Estimator softmax competition and dropout mismatch matter more. |
| 8. Scalability | High | **Medium** | Amplifier, not root cause. At depth 4 this should be salvageable if representation and routing are right. |
| 7. MLP comparison | High | **Framing only** | Correct high-level explanation (smooth dense regression vs sparse combinatorial control) but not an independent hypothesis beyond points 1–5. |

**Priority order: 5 > 1 ≈ 3 > 4 > 2 > 6 > 8**

### Missing from original review

**Training prior mismatch.** The default training prior generates data mostly from MLPs, not from GRANDE trees (`ticl/dataloader.py:91`). The MLP MotherNet therefore has a built-in teacher-student advantage: it learns to fit data that was generated by the same function class it predicts. This alone can explain part of the baseline gap and confounds every comparison between the two paths.

### Revised experiment plan

Ordered by expected information gain per unit effort:

1. **Fix `grande_context_seed` during training** so train and inference use the same tree coordinates. Directly tests point 5 in isolation. If accuracy improves, the coordinate mismatch was silently capping performance.

2. **Equalize init and temperature handling across decoder variants.** Removes confounds from point 3. Specifically: apply `N(0,0.05)` last-layer init to factorized/depthwise, and enable temperature scheduling for all variants.

3. **Run a tiny GRANDE teacher-student ladder: depth=1, E=1, then increase one axis at a time.** If a minimal GRANDE MotherNet can't match a minimal standalone GRANDE on GRANDE-generated data, the problem is in the decoder/representation, not in scaling or routing sparsity. This immediately narrows the search.

4. **Match the training prior.** Train on GRANDE-generated tasks instead of MLP-generated ones, or use a mixed prior. Tests the missing point directly.
