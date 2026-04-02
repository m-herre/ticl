# Training Speed Optimization Audit

*Date: 2026-04-02*

---

## 1. MULTI-GPU & DISTRIBUTED TRAINING

**Status: Already partially implemented, but with gaps.**

DDP wrapping exists at `train.py:172`:
```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
```

Gradient sync is correctly gated (`train.py:71-74`) via `model.no_sync()` for gradient accumulation steps.

### Finding 1.1 — No `DistributedSampler` on the data loader
**Severity: 🟢 Low (not applicable)**

`PriorDataLoader` generates synthetic data on-the-fly. No `DistributedSampler` needed — each rank generates independent random batches. This is fine.

### Finding 1.2 — No FSDP or tensor parallelism
**Severity: 🟡 Medium**

The model is not sharded. For the current MotherNet size (transformer backbone ~50-100M params), DDP is appropriate. FSDP would only help if model size grows significantly. **No action needed now.**

---

## 2. DATA LOADING & PREPROCESSING BOTTLENECKS

### Finding 2.1 — Python loop over batch elements in MLPPrior
**File:** `ticl/priors/mlp.py:101-105` **Severity: 🔴 High**

```python
# CURRENT: Sequential loop creating one MLP per batch element
sample = [MLP(device, num_features, num_outputs, n_samples, **sample_distributions(self.config))
          .to(device)() for _ in range(batch_size)]
x, y = zip(*sample)
y = torch.cat(y, 1).detach().squeeze(2)
x = torch.cat(x, 1).detach()
```

Each batch element creates a fresh `nn.Module`, moves it to GPU, runs forward, then concatenates. For `batch_size=32`, this is 32 serial GPU kernel launches.

**Fix:** This is architecturally inherent — each MLP has different sampled hyperparameters (depth, width, activation). Batching is difficult but the MLP forward passes themselves are small. The main overhead is Python object creation, not GPU compute. **Estimated impact: ~5-10% of data generation time.**

### Finding 2.2 — Nested Python loops in ClassificationAdapter
**File:** `ticl/priors/classification_adapter.py:244-259` **Severity: 🟡 Medium**

```python
# CURRENT: Up to 10 reshuffles per batch element to ensure train/test class compatibility
for b in range(y.shape[1]):
    is_compatible, N = False, 0
    while not is_compatible and N < 10:
        targets_in_train = torch.unique(y[:single_eval_pos, b], sorted=True)
        targets_in_eval = torch.unique(y[single_eval_pos:, b], sorted=True)
        is_compatible = (len(targets_in_train) == len(targets_in_eval)
                         and (targets_in_train == targets_in_eval).all()
                         and len(targets_in_train) > 1)
        if not is_compatible:
            randperm = torch.randperm(x.shape[0])
            x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
        N = N + 1
```

Each `torch.unique()` call triggers a GPU sync. With `batch_size=32` and up to 10 retries, that's up to **640 GPU syncs** in the worst case.

**Fix:** Vectorize the compatibility check across the batch dimension:

```python
# AFTER: Vectorized — compute unique counts in one pass
for _ in range(10):  # max retries
    train_uniq = [torch.unique(y[:single_eval_pos, b]) for b in range(y.shape[1])]
    eval_uniq = [torch.unique(y[single_eval_pos:, b]) for b in range(y.shape[1])]
    incompatible = torch.tensor([
        not (len(t) == len(e) and len(t) > 1 and (t == e).all())
        for t, e in zip(train_uniq, eval_uniq)
    ])
    if not incompatible.any():
        break
    # Reshuffle only incompatible batch elements
    randperm = torch.randperm(x.shape[0])
    x[:, incompatible] = x[randperm][:, incompatible]
    y[:, incompatible] = y[randperm][:, incompatible]
```

**Estimated speedup: 2-5x for this section** (reduces from `O(batch * retries)` syncs to `O(retries)` syncs).

### Finding 2.3 — Categorical feature loop
**File:** `ticl/priors/classification_adapter.py:204-209` **Severity: 🟡 Medium**

```python
for col in range(x.shape[2]):  # iterates over ALL features
    num_unique_features = max(round(random.gammavariate(1, 10)), 2)
    m = MulticlassRank(num_unique_features, ordered_p=0.3)
    if random.random() < p:
        x[:, :, col] = m(x[:, :, col])
```

Creates a new `MulticlassRank` per feature. With `num_features=100`, that's 100 Python object allocations + per-feature GPU ops. **Can be batched but only matters when `categorical_feature_p > 0`.**

---

## 3. MIXED PRECISION & MEMORY OPTIMIZATIONS

### Finding 3.1 — Mixed precision is implemented
**File:** `ticl/train.py:76` **Severity: 🟢 (already present)**

Autocast and GradScaler are correctly used:
```python
with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if scaler is not None else nullcontext():
```

### Finding 3.2 — GRU forced to fp32, defeating mixed precision for depthwise decoder
**File:** `ticl/models/decoders.py:1123-1135` **Severity: 🟡 Medium**

```python
# CURRENT: Forces fp32 for GRUCell because fused kernel doesn't support bf16
autocast_context = torch.autocast(device_type="cuda", enabled=False) if ...
with autocast_context:
    child_states = self.depthwise_state_cell(
        gru_input_flat.float(),       # <- explicit fp32 cast
        parent_hidden_flat.float(),
    )
child_states = child_states.to(dtype=dtype)  # cast back
```

The GRUCell runs in fp32 regardless of mixed precision. This is a known PyTorch limitation. **Consider replacing GRUCell with a custom linear + sigmoid + tanh equivalent that works in bf16**, or switch to a simple MLP-based state update.

### Finding 3.3 — `.float()` cast in `flatten_grande_estimator_outputs`
**File:** `ticl/models/grande_core.py:177` **Severity: 🟢 Quick win**

```python
# CURRENT
return torch.cat([part.float() for part in parts], dim=-1)
```

Forces all parts to fp32 before concatenation. If this runs inside autocast, it defeats the purpose.

**Fix:**
```python
return torch.cat([part.to(dtype=parts[0].dtype) for part in parts], dim=-1)
```

### Finding 3.4 — No gradient checkpointing
**Severity: 🟡 Medium**

The transformer backbone (`TransformerEncoderSimple`) has no `torch.utils.checkpoint` wrapping. For deep transformers with long sequences, this trades compute for memory, enabling larger batch sizes.

**Fix (in `mothernet.py`, around the transformer encoder):**
```python
# In TransformerEncoderSimple.forward():
from torch.utils.checkpoint import checkpoint
for layer in self.layers:
    output = checkpoint(layer, output, use_reentrant=False)
```

**Estimated impact:** Reduces peak memory ~30-40%, allows ~1.5x batch size at the cost of ~20% more compute. Net effect depends on whether you're memory- or compute-bound.

---

## 4. TRAINING LOOP MICRO-OPTIMIZATIONS

### Finding 4.1 — `optimizer.zero_grad()` without `set_to_none=True`
**File:** `ticl/train.py:109` **Severity: 🟢 Quick win**

```python
# CURRENT
optimizer.zero_grad()

# AFTER
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` avoids a memset operation on every parameter gradient. For AdamW with many parameters, this saves ~1-2% per step.

### Finding 4.2 — Missing `cudnn.benchmark = True`
**Severity: 🟢 Quick win**

No `torch.backends.cudnn.benchmark = True` is set anywhere in MotherNet training (it's commented out in `grande.py:1973`). Since input sizes are fixed per config, enabling this gives cuDNN time to autoselect the fastest algorithms.

**Fix (in `ticl/fit_model.py`, after device init):**
```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

**Estimated speedup: 5-15% for transformer forward/backward.**

### Finding 4.3 — Missing `set_float32_matmul_precision('high')`
**Severity: 🟢 Quick win**

The standalone `grande.py:57` sets `torch.set_float32_matmul_precision('high')` but MotherNet training does not.

**Fix (in `ticl/fit_model.py`):**
```python
torch.set_float32_matmul_precision('high')
```

This enables TF32 on Ampere+ GPUs for fp32 matmuls, giving ~3x speedup on matmuls with negligible precision loss.

### Finding 4.4 — `.cpu().detach().item()` ordering
**File:** `ticl/train.py:114` **Severity: 🟢 Quick win**

```python
# CURRENT
total_loss += loss.mean().cpu().detach().item()

# AFTER (slightly cleaner, avoids extra copy)
total_loss += loss.mean().detach().item()
```

`.item()` already moves to CPU and extracts Python scalar. The explicit `.cpu()` is redundant.

---

## 5. MODEL ARCHITECTURE QUIRKS

### Finding 5.1 — `batch_first=False` disables Flash Attention
**File:** `ticl/models/mothernet.py:510` **Severity: 🔴 High impact**

```python
# CURRENT (mothernet.py:500-510)
# mothernet has batch_first=False, unlike all the other models.
def encoder_layer_creator():
    return TransformerEncoderLayer(
        emsize, nhead, nhid, dropout,
        activation=activation, pre_norm=pre_norm,
        recompute_attn=recompute_attn,
        batch_first=False,  # <- THIS DISABLES FLASH ATTENTION
    )
```

The `layer.py:103-104` comment explicitly says:
```python
# batch_first is set to True for using flash attention II
```

Yet MotherNet overrides it to `False`. PyTorch's `nn.MultiheadAttention` only dispatches to the fast `scaled_dot_product_attention` kernel (Flash Attention v2 / memory-efficient attention) when `batch_first=True`.

**Fix:** Switch to `batch_first=True` and add the necessary transposes at the MotherNet boundary:

```python
# In mothernet.py encoder_layer_creator:
batch_first=True,

# In MotherNet.inner_forward(), add transpose before/after:
# Before: x shape is (seq, batch, emsize)
x = x.transpose(0, 1)           # -> (batch, seq, emsize)
output = self.transformer_encoder(x)
output = output.transpose(0, 1)  # -> (seq, batch, emsize)
```

**Estimated speedup: 1.5-3x for the transformer backbone**, which is a major portion of forward+backward time. This is the single highest-impact change.

**Caveat:** Flash SDP is already disabled on non-SM80/SM90 GPUs at `train.py:160`. On A100/H100, this change enables Flash Attention. On A40 (SM86), it would still use the memory-efficient kernel, which is also faster than the default.

### Finding 5.2 — No `torch.compile()` on the model
**File:** Entire codebase **Severity: 🔴 High impact**

`torch.compile()` is used in standalone `grande.py:624` but **never** in MotherNet training.

**Blockers for full-model compile:**
1. **Depthwise decoder** (`decoders.py:1038-1141`): Loop with data-dependent shapes (`nodes_at_depth` doubles each iteration) — **graph break**.
2. **`.item()` call** (`decoders.py:943`): `self.split_temperature_step.item()` — **CPU sync / graph break**.
3. **Dynamic `single_eval_pos`**: Changes each batch — causes recompilation.

**Fix (partial — compile the hot path only):**

```python
# Compile grande_forward separately — it has static shapes
grande_forward_compiled = torch.compile(grande_forward, mode="reduce-overhead")

# In MotherNet.forward(), use compiled version:
grande_out = grande_forward_compiled(x=x[single_eval_pos:], ...)
```

The `grande_forward()` kernel in `grande_core.py:294-374` has **fixed tensor shapes** (given fixed config) and no data-dependent control flow — it's an excellent `torch.compile()` target.

**Estimated speedup: 10-30% for the grande_forward kernel** (fuses einsum chains, eliminates intermediate allocations).

**Also fix the `.item()` blocker:**
```python
# BEFORE (decoders.py:943)
progress = min(float(self.split_temperature_step.item()) / float(...), 1.0)

# AFTER — avoid .item(), use tensor division
progress = torch.clamp(self.split_temperature_step / self.grande_split_temperature_anneal_steps, max=1.0)
```

### Finding 5.3 — `build_grande_context()` creates tensors on CPU then moves to GPU
**File:** `ticl/models/grande_core.py:131-149` **Severity: 🟢 Quick win**

```python
# CURRENT: allocated on CPU, moved to GPU at return
features_by_estimator = torch.zeros(batch_size, n_estimators, selected_variables, dtype=torch.long)
feature_mask = torch.zeros(batch_size, n_estimators, selected_variables, dtype=torch.bool)
...
return {
    "features_by_estimator": features_by_estimator.to(device),
    "feature_mask": feature_mask.to(device),
}
```

**Fix:**
```python
features_by_estimator = torch.zeros(
    batch_size, n_estimators, selected_variables, dtype=torch.long, device=device
)
feature_mask = torch.zeros(
    batch_size, n_estimators, selected_variables, dtype=torch.bool, device=device
)
```

Eliminates one CPU->GPU transfer per forward pass.

### Finding 5.4 — `path_identifier_list.to(dtype=dtype)` every forward pass
**File:** `ticl/models/grande_core.py:344` **Severity: 🟢 Quick win**

```python
# CURRENT: converts buffer dtype every single forward call
path_ids = path_identifier_list.to(dtype=dtype)
```

This buffer is created once and never changes. Pre-cast it at registration time or cache the cast:

```python
# In __init__ or where buffer is registered:
self.register_buffer('path_ids_float', path_identifier_list.to(dtype=torch.float32))
```

### Finding 5.5 — `gather_estimator_features` double permute
**File:** `ticl/models/grande_core.py:155-160` **Severity: 🟡 Medium**

```python
def gather_estimator_features(x, features_by_estimator):
    x_perm = x.permute(1, 0, 2)                           # (seq, batch, feat) -> (batch, seq, feat)
    batch_index = torch.arange(x_perm.shape[0], device=x.device)[:, None, None]
    gathered = x_perm[batch_index, :, features_by_estimator]  # advanced indexing
    return gathered.permute(3, 0, 1, 2)                      # back to (sel_vars, batch, seq, est)
```

Two permutes + advanced indexing. Called in both `grande_forward()` and `build_grande_feature_stats()`. Consider using `torch.gather` with pre-expanded index tensors, or restructure the memory layout to avoid permutes.

---

## 6. EXPERIMENT & LOGGING OVERHEAD

### Finding 6.1 — Profiling adds 8 `cuda.synchronize()` calls per forward pass
**File:** `ticl/models/decoders.py:1178-1214`, `ticl/models/mothernet.py:64-77` **Severity: 🟡 Medium (when `--grande-profile True`)**

Each profiled section calls `torch.cuda.synchronize()` at both start and end:

```python
def _start_grande_timer(self, x):
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)    # sync 1
    return time.perf_counter()

def _stop_grande_timer(self, x, start):
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)    # sync 2
    return time.perf_counter() - start
```

With 4 timed sections, that's **8 GPU pipeline stalls** per batch. Each sync can cost 0.1-10ms depending on GPU load.

**Fix:** Use CUDA events instead of wall-clock time:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... operation ...
end.record()
torch.cuda.synchronize()  # only once at epoch end
elapsed = start.elapsed_time(end)
```

Or simply disable profiling for production training.

### Finding 6.2 — Checkpoint saving is synchronous
**File:** `ticl/utils.py` (via `make_training_callback`) **Severity: 🟢 Low**

Checkpoints are saved every `save_every` epochs (default 10). At ~4 min/epoch, a blocking save every 40 minutes is acceptable. **No action needed unless `save_every` is set very low.**

### Finding 6.3 — Validation blocks training
**File:** `ticl/utils.py:509-544` **Severity: 🟢 Low**

Validation runs synchronously every `save_every` epochs. Since it's infrequent, the impact is small.

---

## Ranked Action Plan: Do These 5 Things First

| Rank | Finding | File | Change | Est. Speedup |
|------|---------|------|--------|-------------|
| **1** | **Switch to `batch_first=True`** to enable Flash Attention / memory-efficient attention | `mothernet.py:510` | Change to `batch_first=True`, add transposes at MotherNet boundary | **1.5-3x on transformer backbone** |
| **2** | **`torch.compile(grande_forward)`** — the kernel has static shapes and no graph breaks | `grande_core.py:294` + call site in `mothernet.py` | Wrap `grande_forward` with `torch.compile(mode="reduce-overhead")` | **10-30% on forward kernel** |
| **3** | **Set `cudnn.benchmark = True`** + **`set_float32_matmul_precision('high')`** | `fit_model.py` (add 2 lines) | Two lines at startup | **5-15% overall** (TF32 matmuls + cuDNN autotuning) |
| **4** | **`optimizer.zero_grad(set_to_none=True)`** | `train.py:109` | One-word change | **1-2% per step** |
| **5** | **Allocate context tensors directly on GPU** + **pre-cast `path_identifier_list`** | `grande_core.py:131-135, 344` | Add `device=device` to `torch.zeros`, cache dtype cast | **Eliminates CPU->GPU transfers per batch** |

**Bonus (medium-effort, high-reward):**
- **5b.** Fix `.item()` in temperature annealing (`decoders.py:943`) to unblock future `torch.compile()` of the decoder.
- **5c.** Vectorize the train/test compatibility loop in `classification_adapter.py:244-259` to eliminate per-batch-element GPU syncs.
- **5d.** Add gradient checkpointing to transformer layers if memory-bound (enables larger batch sizes).

Items 1-5 can all be implemented in under an hour combined. Item 1 alone (Flash Attention) is likely worth more than all the others put together for long-sequence workloads.
