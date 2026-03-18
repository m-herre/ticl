# Why Predicting GRANDE Parameters is Fundamentally Harder Than MLP Parameters

## And What to Do About It

---

## Part 1: MLP vs GRANDE Parameter Space Comparison

### 1.1 Why MLP Parameters Are "Easy" to Predict

**Smooth parameter→function mapping.** For an MLP with parameters θ, the mapping θ → f_θ(x) is Lipschitz continuous everywhere (outside measure-zero degenerate points). A small perturbation δθ produces a bounded perturbation in the output: ‖f_{θ+δθ}(x) - f_θ(x)‖ ≤ L‖δθ‖. This means the decoder can make small errors in every predicted weight and still get a reasonable function. The loss landscape seen by the MotherNet decoder is smooth with respect to the child MLP's parameters.

**Low-rank structure absorbs symmetry.** Raw MLPs have neuron permutation symmetry: swapping two neurons in a hidden layer (with corresponding input/output weights) yields the same function. This creates a discrete symmetry group of size (h!)^L for L layers of width h, which would make parameter prediction ill-defined — the same function has exponentially many parameter representations. But MotherNet's low-rank decomposition W = W_pred × W_fixed *breaks this symmetry*. The fixed matrices W_f (learned during meta-training) select a canonical basis for each hidden layer. The decoder only predicts W_pred ∈ R^{h×r}, and since W_f is fixed, there is essentially a unique W_pred for each desired weight matrix W. This is a crucial and underappreciated design feature.

**Dense, distributed participation.** Every weight in an MLP participates in computing every output for every input. Errors are averaged across dimensions. If the decoder gets one entry of W_pred wrong by ε, this corrupts one rank-1 component of W, spreading the error across all h neurons proportionally. No single parameter has outsized influence on the function.

**Additive layer composition.** Each MLP layer computes h_l = relu(W_l · h_{l-1} + b_l). The contribution of each weight to the output is *additive* (through the linear map) before the nonlinearity. While ReLU introduces piecewise structure, the activation regions change continuously with the weights. There are no hard combinatorial switches.

**Flat output structure.** The decoder outputs a vector φ that is simply reshaped into matrices. There are no internal constraints between different parts of φ — any real-valued vector is a valid parameter configuration. The decoder never needs to satisfy consistency conditions.

### 1.2 Why GRANDE Parameters Are Hard to Predict

The GRANDE parameter space has fundamentally different geometry that makes one-shot prediction by a standard decoder pathological.

#### 1.2.1 Discrete–Continuous Entanglement

GRANDE has four parameter groups with coupled discrete and continuous semantics:

| Parameter | Type | Shape per estimator | Role |
|-----------|------|---------------------|------|
| `split_index_logits` (I) | Discrete (after ST hardmax) | (n_nodes, n_features_local) | Which feature to split on |
| `split_values` (T) | Continuous | (n_nodes,) | Threshold value for the selected feature |
| `leaf_classes` (L) | Continuous | (n_leaves, n_out) | Class logits at each leaf |
| `estimator_weights` (W) | Continuous | (n_leaves,) | Instance-wise importance per leaf |

The critical coupling: **the threshold T at node j is only meaningful in the coordinate system of the feature selected by I at node j.** If the decoder predicts I_logits such that feature 3 is selected in the hard forward pass, but the gradient (through the soft backward pass) partially attributes to feature 7, then:

- The forward loss is computed using threshold T compared against feature 3's values
- The backward gradient for T incorporates feature 7's statistics (weighted by softmax probability)
- The decoder receives a training signal that is a *mixture* of what T should be for feature 3 and what it should be for feature 7

These features may have completely different scales (feature 3 in [0,1], feature 7 in [-100, 100]). This forward–backward mismatch creates a noisy, inconsistent gradient signal for the decoder. In standalone GRANDE, this is manageable because the parameters are updated incrementally over many gradient steps and can track the coupling as it evolves. In a hypernetwork, the decoder must get the *joint configuration* right in one shot.

#### 1.2.2 Multiplicative Path Composition Creates Catastrophic Sensitivity

In `tree_forward`, a leaf's activation is a **product** over depth:

```
L(x|l) = ∏_{j=1}^{d} [(1-p(l,j)) · S_j + p(l,j) · (1-S_j)]
```

where S_j ∈ {0, 1} after ST rounding. This product is the key structural difference from MLPs. Consider what happens when the decoder makes a small error:

- **MLP**: An error ε in one weight perturbs the output by O(ε) (through linear propagation + bounded nonlinearity).
- **Tree**: An error in a single split parameter at depth 1 can flip S_1 from 0→1 for some samples. This redirects those samples to an *entirely different subtree*. All splits at depths 2...d and the leaf values in the original subtree become irrelevant; the samples now hit a different leaf with potentially unrelated class logits.

This is not a smooth degradation — it is a **discrete phase transition** in the function. The parameter→function map has discontinuities along hyperplanes in parameter space (where a feature value equals a threshold for some training point). The decoder must predict parameters that avoid these catastrophic boundaries, but the boundaries themselves depend on the input data, creating a complex, data-dependent constraint manifold.

#### 1.2.3 Enormous Discrete Symmetry Group

**Estimator permutation**: For E estimators, permuting any two yields the same ensemble (up to relabeling). This gives E! equivalent parameterizations. With E = 2048 (as used in GRANDE), this is 2048! ≈ 10^{5900} — an incomprehensibly large symmetry group. 

But it is worse than simple permutation:

- **Subtree reflection**: In a binary tree, swapping the left and right children of any internal node (and flipping the corresponding split direction) yields the same function. For a tree of depth d, there are 2^{2^d - 1} such reflections per estimator.
- **Dead subtree equivalence**: If a split at depth j sends all training data to one child, the entire opposite subtree is unconstrained — any parameters there yield the same loss. This creates flat directions in the loss landscape that vary per dataset.
- **Feature redundancy**: If two features are correlated, selecting either one at a split node can produce a similar function, creating near-symmetries that the decoder cannot resolve.

For MLPs, the low-rank structure W_pred × W_fixed breaks neuron permutation symmetry. **There is no analogous symmetry-breaking mechanism in the current GrandeDecoder.** The decoder must somehow learn to always output estimators in a "canonical" order, but nothing in the architecture encourages this. During meta-training, different synthetic datasets may have their optimal estimators in different orders, creating contradictory gradients.

#### 1.2.4 Non-Independent Parameters (Compositional Constraints)

In an MLP, the three weight matrices (W_1, W_2, W_3) are compositional but each is independently valid as a linear map — any real matrix works. In a GRANDE tree:

- The *meaning* of leaf parameters L depends on which samples reach that leaf, which depends on ALL splits above it.
- The *correct threshold* at a child node depends on the data distribution AFTER the parent split has partitioned the data.
- The *correct estimator weights* depend on which leaf each sample reaches, which depends on all splits.

These are not soft dependencies — they are hard structural constraints. The parameter vector is not a product space; it is a highly constrained manifold where each coordinate's valid range depends on other coordinates.

#### 1.2.5 Scale Heterogeneity

The four parameter types live in very different numerical regimes:

- **I_logits**: Pre-softmax logits; typically in [-5, 5] to produce peaked selections. The *differences* between logits matter, not absolute values.
- **T (thresholds)**: In the data space, determined by feature statistics. If features are quantile-transformed to [0,1], thresholds are in [0,1]. If not, they can be arbitrary.
- **L (leaf classes)**: Pre-softmax class logits; scale determines confidence.
- **W (leaf weights)**: Pre-softmax weights; the *relative* values across estimators matter.

An MLP decoder outputting all four as a flat vector must somehow learn to use different output neurons at different scales for different roles. This is possible in principle but creates optimization difficulty — the gradients from the tree loss back to different parameter groups have vastly different magnitudes and dynamics.

---

## Part 2: Failure Mode Analysis

### 2.1 Why a Standard MLP Decoder Systematically Fails

The current architecture (GrandeDecoder/GradTreeDecoder) works as:

```
SummaryLayer(transformer_output) → E ∈ R^{m_all}
E → MLP_decoder → φ ∈ R^{total_params}
reshape(φ) → (I, T, L, W)
```

This is essentially the same architecture as MLPModelDecoder, just with a different reshape at the end. Here is why this fails specifically for tree parameters:

**Failure Mode 1: Feature Selection Averaging.** The MLP decoder has a well-known spectral bias — it preferentially represents low-frequency functions. Applied to the map from dataset embeddings to split_index_logits, this means the decoder tends to output I_logits that are *smooth functions* of the dataset embedding. Since different datasets require different feature selections, and the decoder averages over many training distributions, the I_logits tend to be insufficiently peaked. They end up close to uniform, making the ST hardmax selection noisy and effectively random at early stages of training. This creates a chicken-and-egg problem: the thresholds can't be learned until feature selection stabilizes, but feature selection can't stabilize until the gradients (which flow through thresholds) are meaningful.

**Failure Mode 2: Threshold Predictions Ignore Feature Identity.** The decoder predicts thresholds T as a generic function of the dataset embedding, without explicitly knowing which feature was selected at each node. But the *correct* threshold at node j depends critically on the distribution of the *selected feature* at node j, conditioned on the data that reaches node j (which depends on all parent splits). The MLP decoder has no mechanism to perform this conditional computation. It can only learn a statistical average over possible feature selections, which is wrong for any specific selection.

**Failure Mode 3: No Mechanism for Estimator Diversification.** The MLP decoder maps one embedding E to ALL estimator parameters simultaneously. Since the same E is the sole input, the decoder has no built-in way to make different estimators different — it must learn to carve E into E distinct estimator parameterizations using the network weights alone. This is a hard combinatorial assignment problem embedded in a smooth optimization. The typical outcome is that many estimators collapse to similar trees, wasting ensemble capacity.

**Failure Mode 4: Flat Vector Ignores Hierarchical Dependencies.** The decoder outputs a flat vector φ and reshapes it. This treats the threshold at (estimator=3, depth=1, position=0) — a root split — as having the same status as (estimator=3, depth=4, position=7) — a deep leaf-adjacent split. But root splits are *far more important*: getting the root split wrong makes the entire tree wrong, while getting a deep split wrong only affects a small fraction of samples. The MLP decoder allocates equal capacity to all parameters, when it should concentrate capacity on structurally important ones.

**Failure Mode 5: Gradient Magnitude Collapse for Deep Splits.** During backpropagation through `tree_forward`, the gradient of the loss with respect to a split parameter at depth j passes through a product of j terms (the path probability). As j increases, this product shrinks exponentially (each term is ≤ 1), creating vanishing gradients for deep splits. The decoder receives strong gradients for root splits and near-zero gradients for deep splits, making it unable to learn correct deep split parameters. In standalone GRANDE, this is mitigated by long training with adaptive learning rates; a one-shot decoder has no such luxury.

### 2.2 Specific Error Patterns

Based on the above analysis, the decoder is likely making these specific errors:

1. **Split index logits are too uniform** → feature selection is noisy, especially at deeper levels
2. **Thresholds are biased toward feature-averaged values** → splits are often near the median of the *wrong* feature
3. **Estimators are near-copies of each other** → effective ensemble size << nominal ensemble size
4. **Deep tree nodes are essentially random** → the ensemble acts as if it has effective depth << specified depth
5. **Leaf weights are near-uniform** → instance-wise weighting degrades to simple averaging, losing GRANDE's key advantage

---

## Part 3: Decoder Design Proposals

### Proposal A: Factorized Decoder with Feature-Conditioned Thresholds

**Core insight**: Separate discrete structure prediction from continuous value prediction, and condition values on the chosen structure using data statistics.

**Architecture**:

```
                    ┌─────────────────┐
  Dataset Embedding │  Structure Head  │──→ I_logits (all nodes, all estimators)
  E ∈ R^{m_all}    │  (MLP_struct)    │         ↓ ST hardmax
                    └─────────────────┘    I_hard (one-hot)
                                               ↓
                    ┌─────────────────┐    Feature Stats
                    │  Threshold Head  │←── μ_f, σ_f per node
                    │  (MLP_thresh)    │    (gathered from training data
                    └─────────────────┘     using I_hard)
                           ↓
                    T = μ_f + MLP_thresh(E, I_hard, μ_f, σ_f) × σ_f
                    
                    ┌─────────────────┐
                    │   Leaf Head      │──→ L (leaf class logits)
                    │  (MLP_leaf)      │──→ W (leaf estimator weights)
                    └─────────────────┘
```

**Step-by-step**:

1. **Structure Head** (`MLP_struct`): Takes E, outputs `split_index_logits` for all (estimator, node) pairs. Shape: `(batch, n_estimators, n_nodes, n_features_local)`. Uses Gumbel-Softmax during training (temperature annealed from τ=1.0 to τ=0.1 over meta-training) instead of vanilla softmax+ST. This provides lower-variance gradient estimates for discrete selections.

2. **Feature statistics gathering**: Given the hard feature selections `I_hard`, gather the mean μ_f and standard deviation σ_f of the selected feature from the training data for each node. This is a differentiable gather operation (straight-through through the argmax). *This data is already partially available from `build_grande_context()` which computes per-estimator feature statistics.*

3. **Threshold Head** (`MLP_thresh`): Takes `[E, I_hard (embedded), μ_f, σ_f]` and outputs a *normalized residual* δ per node. The actual threshold is `T = μ_f + δ × σ_f`. This reparameterization means the decoder only needs to predict *where within the feature's distribution to split* — a problem that is roughly on the same scale ([-2, 2] standard deviations) regardless of the feature.

4. **Leaf Head** (`MLP_leaf`): Takes E and outputs leaf class logits L and leaf weights W. These can remain as simple MLP predictions since they are purely continuous and don't suffer from the discrete-continuous coupling.

**Why this matches GRANDE**:
- Respects the discrete-continuous duality that is the core structural property of trees
- Feature statistics conditioning gives the threshold predictor the right "coordinate system" — it no longer needs to implicitly learn feature distributions
- The Gumbel-Softmax provides a proper relaxation of the discrete feature selection, with a principled annealing schedule

**Why this improves learning dynamics**:
- The threshold reparameterization `T = μ + δσ` makes the threshold prediction task **feature-scale invariant**. The MLP_thresh always outputs values in roughly [-3, 3] regardless of the actual feature range. This eliminates the scale heterogeneity problem.
- Gumbel-Softmax gradients have lower variance than ST gradients for categorical variables, especially early in training when the logits are not yet peaked.
- The factorization means the structure head and threshold head receive *different* gradient signals — the structure head gets gradients about *which* feature to split on, while the threshold head gets gradients about *where* to split. Currently these are entangled in a single MLP.

**Implementation cost**: Moderate. Requires modifying `GrandeDecoder` to have separate heads and adding the feature statistics gathering step. The `build_grande_context()` already computes per-estimator feature statistics, so this infrastructure partially exists. The Gumbel-Softmax is a drop-in replacement for softmax+ST.

---

### Proposal B: Top-Down Autoregressive Tree Decoder

**Core insight**: Generate tree parameters depth-by-depth, conditioning child splits on parent decisions. This respects the compositional structure where the *meaning* of a split at depth j depends on all splits at depths 1...(j-1).

**Architecture**:

```
  Dataset Embedding E ∈ R^{m_all}
         ↓
  ┌──────────────────────────────────┐
  │  Estimator Embedding Layer       │
  │  E → [e_1, ..., e_E] ∈ R^{E×d} │
  │  (Linear + reshape)              │
  └──────────────────────────────────┘
         ↓
  For each depth j = 1, ..., d:
  ┌──────────────────────────────────────────────────────┐
  │  TreeGRU(e_i, parent_state_{j-1}) → node_state_j    │
  │                                                       │
  │  SplitHead(node_state_j) → I_logits_j, T_j          │
  │                                                       │
  │  parent_state_j = f(node_state_j, I_hard_j, T_j)    │
  │  (state includes: which feature selected,             │
  │   threshold value, left/right branch indicator)       │
  └──────────────────────────────────────────────────────┘
         ↓
  After all depths:
  ┌──────────────────────────────────┐
  │  LeafHead(e_i, path_states)      │
  │  → L_i, W_i                      │
  └──────────────────────────────────┘
```

**Step-by-step**:

1. **Estimator Embedding**: Map E to E estimator-level embeddings e_1,...,e_E. Each is a d_est-dimensional vector. This provides each estimator with a unique "seed" that enables diversification.

2. **Depth-wise generation**: For depth j from 1 to d (the tree depth):
   - Each node at depth j receives: (a) the estimator embedding e_i, (b) a parent state encoding which ancestor split was taken and what feature/threshold was used.
   - A small GRU cell (shared across depths, estimators, and nodes at the same depth) updates the state:  `h_j = GRU(h_{j-1}, [e_i, depth_emb_j])`
   - A split head MLP maps h_j → (I_logits_j, T_j) for this node.
   - After ST hardmax on I_logits_j, the hard feature selection and threshold are encoded back into the state for the children.

3. **Leaf generation**: After all splits are determined, each leaf receives the path state (encoding the full root→leaf path of splits). A leaf head MLP maps this to (leaf_class_logits, leaf_weight).

4. **Parallelism**: Within each depth level, all 2^{j-1} nodes across all E estimators can be processed in parallel. The sequential bottleneck is only d steps (typically 4–6). With batched operations over (batch × E × nodes_at_depth), this is efficient on GPU.

**Why this matches GRANDE**:
- The top-down generation mirrors how decision trees actually partition data — the root split determines the two subsets that child nodes operate on.
- Conditioning child splits on parent decisions captures the key dependency: "if the root splits on feature 3 at value 0.5, the left child should consider features that are informative *within the subset x_3 < 0.5*."
- Leaf parameters are generated after the full tree structure is known, so they are in the correct context.

**Why this improves learning dynamics**:
- **Root splits get priority**: The first GRU step generates root splits with fresh, unattenuated gradients. Deep splits are generated later, but their gradients still only pass through the shallow GRU, not through the product of path probabilities. This is a form of implicit gradient highway.
- **Conditional structure is learnable**: The GRU can learn patterns like "after splitting on feature A, the child should split on feature B" — these conditional dependencies are precisely what standalone GRANDE learns over many gradient steps but what a flat MLP decoder cannot express.
- **Estimator diversity is natural**: Different estimator embeddings e_i feed into the same GRU with different initial conditions, naturally producing different trees without requiring the decoder to learn a complex combinatorial assignment.

**Implementation cost**: Higher than Proposal A. Requires replacing the single MLP decoder with a depth-recurrent architecture. The GRU has small hidden state (e.g., 256), so the parameter count is modest. The d sequential steps add latency but d ≤ 6 is negligible compared to the transformer backbone.

---

### Proposal C: Slot-Based Estimator Decoder with Cross-Attention

**Core insight**: Use learnable estimator "slots" that attend to the per-sample transformer embeddings, enabling each estimator to specialize on different data regions — matching GRANDE's instance-wise weighting.

**Architecture**:

```
  Per-sample embeddings from transformer: Z ∈ R^{n_train × batch × emsize}
  Learnable estimator slots: S ∈ R^{E × d_slot}  (fixed across datasets, learned in meta-training)
  
  For k = 1, ..., K refinement rounds:
  ┌─────────────────────────────────────────────────────┐
  │  1. Cross-attention: S_k = CrossAttn(Q=S_{k-1},    │
  │                                       K=Z, V=Z)     │
  │     Each slot queries the dataset embeddings         │
  │                                                      │
  │  2. Slot self-attention: S_k = SelfAttn(S_k)        │
  │     Estimators "see" each other → diversification    │
  │                                                      │
  │  3. Slot FFN: S_k = FFN(S_k)                        │
  │     Per-slot nonlinear update                        │
  └─────────────────────────────────────────────────────┘
         ↓
  Parameter Readout per estimator:
  ┌─────────────────────────────────┐
  │  MLP_split(s'_i) → I_i, T_i    │
  │  MLP_leaf(s'_i)  → L_i, W_i    │
  └─────────────────────────────────┘
```

**Step-by-step**:

1. **Slot initialization**: E learnable slot vectors S ∈ R^{E × d_slot}, initialized randomly, learned during meta-training. These are analogous to MotherNet's W_fixed matrices — they provide a canonical basis that breaks estimator permutation symmetry.

2. **Iterative refinement** (K = 2–4 rounds):
   - *Cross-attention*: Each slot attends over all n_train sample embeddings from the transformer. Attention weights are slot-specific, so different slots can focus on different data subsets. This is the architectural analog of GRANDE's `data_subset_fraction` — instead of random subsetting, each estimator learns to attend to the data region it will specialize in.
   - *Slot self-attention*: Estimator slots attend to each other. This enables anti-correlation: if slot 3 is already attending to data region A, slot 7 can learn to focus elsewhere. This is a diversity mechanism that prevents estimator collapse.
   - *FFN*: Standard feedforward per slot for nonlinear refinement.

3. **Parameter readout**: Each refined slot s'_i is mapped to tree parameters by shared MLPs. Can be combined with the factorized structure from Proposal A (separate structure/threshold heads).

**Why this matches GRANDE**:
- GRANDE's instance-wise weighting means different estimators have different importance for different samples. The cross-attention mechanism lets each estimator slot "choose" its data region, which is the learned analog of GRANDE's data_subset_fraction + instance-wise weighting.
- The slot self-attention provides the anti-correlation mechanism that GRANDE achieves through its end-to-end training over many epochs.
- Learnable slots break the estimator permutation symmetry, solving one of the core identifiability problems.

**Why this improves learning dynamics**:
- **Iterative refinement replaces one-shot prediction**: Complex structured outputs benefit from multi-step generation. Each refinement round can correct errors from the previous round. This is well-established in the slot attention literature (Locatello et al., 2020).
- **Cross-attention is more expressive than summarization**: The current SummaryLayer compresses all data into a single vector before decoding. This is a severe bottleneck — dataset-level statistics may not contain enough information to predict per-estimator parameters. Cross-attention preserves per-sample information and lets each estimator extract what it needs.
- **Permutation symmetry breaking**: The learnable slots establish a canonical ordering of estimators, eliminating the E! symmetry that plagues training.

**Implementation cost**: Highest of the three proposals. Requires replacing SummaryLayer + MLP decoder with a multi-round attention architecture. The cross-attention over n_train samples adds O(E × n_train) computation per round. With E = 10–100 (MotherNet scale, not GRANDE's 2048) and n_train ≤ 3000, this is feasible. Would not scale to E = 2048 without approximations.

---

### Proposal A+B Hybrid: Factorized Autoregressive Decoder (Recommended)

**This is my top recommendation.** It combines the key insights of Proposals A and B:

```
  Dataset Embedding E
         ↓
  Estimator Embeddings [e_1,...,e_E]  (from E via MLP + reshape)
         ↓
  For depth j = 1,...,d:
    ┌────────────────────────────────────────────────┐
    │  h_j = GRU(h_{j-1}, [e_i, depth_emb_j])       │
    │                                                 │
    │  I_logits_j = MLP_struct(h_j)                   │  ← Gumbel-Softmax
    │  I_hard_j = ST_hardmax(I_logits_j)              │
    │                                                 │
    │  μ_j, σ_j = gather_feature_stats(I_hard_j)     │  ← from training data
    │  δ_j = MLP_thresh(h_j, I_hard_j, μ_j, σ_j)    │
    │  T_j = μ_j + δ_j × σ_j                         │
    │                                                 │
    │  h_j = update(h_j, I_hard_j, T_j)              │  ← feed decisions back
    └────────────────────────────────────────────────┘
         ↓
  LeafHead(h_final, e_i) → L_i, W_i
```

This gives you:
- **Factorized structure/value prediction** (from A): feature-conditioned thresholds, Gumbel-Softmax
- **Top-down autoregressive generation** (from B): compositional conditioning, root-split priority
- **Moderate implementation cost**: the GRU is small, the depth loop is short, and most operations are batched over (batch × E × nodes_at_depth)

---

## Part 4: Prioritized Ranking

### Rank 1: Proposal A+B Hybrid (Factorized Autoregressive Decoder)

**Justification**: This addresses the three most impactful failure modes simultaneously:
1. Discrete-continuous entanglement → factorized heads + feature-conditioned thresholds
2. Compositional dependencies → top-down autoregressive generation
3. Feature selection instability → Gumbel-Softmax with temperature annealing

It is architecturally clean, has modest implementation cost, and every component has a clear mechanistic justification. The feature statistics conditioning in particular should give a large immediate improvement because it transforms the threshold prediction from an unconstrained regression problem (predict a number in arbitrary range) to a normalized residual prediction (predict a value in [-3, 3]).

**First test**: Implement Proposal A alone (factorized heads + feature-conditioned thresholds) as the minimal intervention. If this shows improvement, add the autoregressive depth loop (Proposal B).

### Rank 2: Proposal A (Factorized Decoder with Feature-Conditioned Thresholds)

**Justification**: This is the minimal effective intervention. It requires the least architectural change — the GrandeDecoder is split into three heads instead of one, and a feature statistics gathering step is added. The `build_grande_context()` infrastructure already computes per-estimator feature statistics. The Gumbel-Softmax is a drop-in replacement.

This should be implemented first as a baseline for measuring the impact of factorization alone vs. factorization + autoregressive generation.

### Rank 3: Proposal C (Slot-Based Decoder)

**Justification**: Most powerful in principle — the cross-attention mechanism provides the richest conditioning and naturally handles estimator specialization. But it has the highest implementation cost, the most hyperparameters to tune (K rounds, d_slot, attention heads), and does not directly address the compositional depth structure of trees. Best reserved for a second iteration after Proposals A/B have established the factorization baseline.

---

## Part 5: Summary of Key Mechanistic Insights

| Property | MLP | GRANDE | Impact on Hypernetwork |
|----------|-----|--------|----------------------|
| Parameter→function continuity | Lipschitz continuous | Discontinuous (split flips) | Tree decoder loss landscape has cliffs; small parameter errors cause large function errors |
| Symmetry group | h!^L (broken by W_fixed) | E! × 2^{n_nodes × E} (unbroken) | Training signal is contradictory across datasets; decoder cannot find canonical form |
| Parameter independence | Each weight independently valid | Threshold meaning depends on feature selection | Flat decoder cannot capture conditional structure |
| Gradient flow through depth | Additive (bounded attenuation) | Multiplicative (exponential attenuation) | Deep split parameters receive vanishing gradients |
| Scale homogeneity | All weights in similar range | I_logits, T, L, W in different regimes | Single output layer must learn heterogeneous scales |
| Output structure | Flat vector → reshape | Hierarchical tree with depth dependencies | Flat prediction ignores that root matters more than leaves |

The fundamental difficulty is that **tree parameters encode a program, not a weight matrix**. Predicting a program requires understanding its compositional semantics — which the current flat MLP decoder does not have. The proposals above introduce compositional structure into the decoder to match the compositional structure of the target.
