# Training Speed Optimization Audit

*Date: 2026-04-02 — Revised after Codex review*

---

## 1. MULTI-GPU & DISTRIBUTED TRAINING

**Status: Already implemented. No changes needed.**

DDP wrapping exists at `train.py:172`:
```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
```

Gradient sync is correctly gated (`train.py:71-74`) via `model.no_sync()` for gradient accumulation steps. `PriorDataLoader` generates synthetic data on-the-fly, so no `DistributedSampler` is needed. FSDP is unnecessary at current model scale.

---

## 2. DATA LOADING & PREPROCESSING BOTTLENECKS

### Finding 2.1 — Python loop over batch elements in MLPPrior
**File:** `ticl/priors/mlp.py:101-105` **Severity: 🟡 Medium**

```python
sample = [MLP(device, num_features, num_outputs, n_samples, **sample_distributions(self.config))
          .to(device)() for _ in range(batch_size)]
```

Each batch element creates a fresh `nn.Module`, moves it to GPU, runs forward, then concatenates. For `batch_size=32`, this is 32 serial GPU kernel launches. Architecturally inherent — each MLP has different sampled hyperparameters. The main overhead is Python object creation, not GPU compute.

### Finding 2.2 — Nested Python loops in ClassificationAdapter
**File:** `ticl/priors/classification_adapter.py:244-259` **Severity: 🟡 Medium**

```python
for b in range(y.shape[1]):
    is_compatible, N = False, 0
    while not is_compatible and N < 10:
        targets_in_train = torch.unique(y[:single_eval_pos, b], sorted=True)
        targets_in_eval = torch.unique(y[single_eval_pos:, b], sorted=True)
        ...
```

Each `torch.unique()` call triggers a GPU sync. With `batch_size=32` and up to 10 retries, that's up to **640 GPU syncs** in the worst case. A true fix would require a batched unique-counting approach (e.g. sort + diff-based counting across the batch dimension) — the per-element `torch.unique()` calls are the core bottleneck, not just the loop structure.

### Finding 2.3 — Categorical feature loop
**File:** `ticl/priors/classification_adapter.py:204-209` **Severity: 🟡 Medium**

Creates a new `MulticlassRank` per feature. With `num_features=100`, that's 100 Python object allocations + per-feature GPU ops. **Only matters when `categorical_feature_p > 0`.**

---

## 3. MIXED PRECISION & MEMORY OPTIMIZATIONS

### Finding 3.1 — GradScaler constructed but never used (BUG)
**File:** `ticl/train.py:217, 76, 104, 108` **Severity: 🔴 High (correctness bug on fp16 hardware)**

```python
# train.py:217 — scaler is created
scaler = GradScaler() if train_mixed_precision and device != "cpu" else None

# train.py:76 — scaler gates autocast (this works)
with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if scaler is not None else nullcontext():

# train.py:104 — BUT loss scaling is never applied
loss.backward()           # should be: scaler.scale(loss).backward()

# train.py:108 — AND scaler.step/update are never called
optimizer.step()          # should be: scaler.step(optimizer)
                          # missing:   scaler.update()
```

The `scaler` object serves as a boolean gate for autocast but its actual loss-scaling methods are never invoked. This means:
- **Autocast IS active** (forward pass runs in mixed precision) ✓
- **Loss scaling is NOT used** ✗

**Impact:** On bf16-capable GPUs (A100, H100), this is harmless — bf16 has the same exponent range as fp32 and doesn't need loss scaling. On older GPUs that fall back to fp16, gradients may underflow silently, causing slow convergence or instability.

**Fix:**
```python
# Option A: Use scaler properly for fp16 compatibility
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()

if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Option B: If only bf16 GPUs are targeted, remove the scaler entirely
#           and gate autocast with a simple boolean flag instead.
```

### Finding 3.2 — GRU forced to fp32, defeating mixed precision for depthwise decoder
**File:** `ticl/models/decoders.py:1123-1135` **Severity: 🟡 Medium**

```python
# Forces fp32 for GRUCell because fused kernel doesn't support bf16
with autocast_context:
    child_states = self.depthwise_state_cell(
        gru_input_flat.float(),
        parent_hidden_flat.float(),
    )
child_states = child_states.to(dtype=dtype)
```

Known PyTorch limitation. Consider replacing GRUCell with a custom linear + sigmoid + tanh equivalent that works in bf16, or a simple MLP-based state update.

### Finding 3.3 — `.float()` cast in `flatten_grande_estimator_outputs`
**File:** `ticl/models/grande_core.py:177` **Severity: 🟢 Quick win**

```python
# CURRENT — forces fp32
return torch.cat([part.float() for part in parts], dim=-1)

# AFTER — preserve input dtype
return torch.cat([part.to(dtype=parts[0].dtype) for part in parts], dim=-1)
```

### Finding 3.4 — Partial gradient checkpointing already exists; full-layer checkpointing may help
**Severity: 🟡 Medium**

Attention recomputation is already enabled via `recompute_attn=True` (default in `model_configs.py:34`, implemented in `layer.py:195`). This checkpoints the QKV attention computation. Full-layer `torch.utils.checkpoint` wrapping would additionally save FFN and layer-norm intermediate activations, trading ~20% more compute for ~30-40% memory reduction. Worth testing if memory-bound.

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

Avoids a memset operation on every parameter gradient. Saves ~1-2% per step.

### Finding 4.2 — Missing `set_float32_matmul_precision('high')`
**File:** `ticl/fit_model.py` **Severity: 🟢 Quick win**

The standalone `grande.py:57` sets this but MotherNet training does not.

```python
torch.set_float32_matmul_precision('high')
```

Enables TF32 on Ampere+ GPUs for fp32 matmuls — affects the linear layers and einsum operations in the transformer backbone and GRANDE kernel. The actual speedup depends on how much time is spent in fp32 matmuls (vs. bf16 under autocast).

### Finding 4.3 — `.cpu().detach().item()` ordering
**File:** `ticl/train.py:114` **Severity: 🟢 Quick win**

```python
# CURRENT
total_loss += loss.mean().cpu().detach().item()

# AFTER — .item() already moves to CPU
total_loss += loss.mean().detach().item()
```

---

## 5. MODEL ARCHITECTURE QUIRKS

### Finding 5.1 — `batch_first=False` bypasses the MHA fast path
**File:** `ticl/models/mothernet.py:510` **Severity: 🟡 Medium**

```python
batch_first=False,  # MotherNet overrides default True
```

**Clarification:** Both `batch_first=True` and `batch_first=False` paths reach `scaled_dot_product_attention` in PyTorch 2.1+ (verified empirically by Codex review). However, `batch_first=True` is a precondition for `_native_multi_head_attention`, the C++ fast path that bypasses Python-level `multi_head_attention_forward` and avoids extra transposes. The speedup is modest (5-15%), not the 1.5-3x originally claimed.

**Fix:** Switch to `batch_first=True` and add transposes at the MotherNet boundary. Low risk, moderate reward.

### Finding 5.2 — `torch.compile()` on `grande_forward` (with dynamic shapes)
**File:** `ticl/models/grande_core.py:294` **Severity: 🟡 Medium**

The `grande_forward()` kernel is a good compile target (no data-dependent control flow, no `.item()` calls). However, `single_eval_pos` varies per batch, so the test-set sequence dimension `x[single_eval_pos:]` is dynamic.

**Fix:** Use `dynamic=True` (not `mode="reduce-overhead"` which requires static shapes for CUDA graphs):

```python
grande_forward_compiled = torch.compile(grande_forward, dynamic=True)
```

This uses symbolic shapes to avoid recompilation for every new sequence length. Expected speedup: 10-20% on the kernel (fuses einsum chains, eliminates intermediate allocations).

**Also fix the `.item()` blocker in the decoder:**
```python
# BEFORE (decoders.py:943)
progress = min(float(self.split_temperature_step.item()) / float(...), 1.0)

# AFTER
progress = torch.clamp(self.split_temperature_step / self.grande_split_temperature_anneal_steps, max=1.0)
```

### Finding 5.3 — `build_grande_context()` creates tensors on CPU then moves to GPU
**File:** `ticl/models/grande_core.py:131-149` **Severity: 🟢 Quick win**

```python
# CURRENT: allocated on CPU, filled with GPU-sampled data, then moved back to GPU
features_by_estimator = torch.zeros(batch_size, n_estimators, selected_variables, dtype=torch.long)
# ... sampled_features comes from GPU generator ...
features_by_estimator[..., :take] = sampled_features  # implicit GPU->CPU transfer
return {"features_by_estimator": features_by_estimator.to(device)}  # CPU->GPU transfer
```

This causes two unnecessary device transfers per forward pass. Allocate directly on device:

```python
features_by_estimator = torch.zeros(
    batch_size, n_estimators, selected_variables, dtype=torch.long, device=device
)
feature_mask = torch.zeros(
    batch_size, n_estimators, selected_variables, dtype=torch.bool, device=device
)
```

### Finding 5.4 — `path_identifier_list.to(dtype=dtype)` every forward pass
**File:** `ticl/models/grande_core.py:344` **Severity: 🟢 Quick win**

Buffer is static — pre-cast at registration time instead of converting every forward call.

### Finding 5.5 — `gather_estimator_features` double permute
**File:** `ticl/models/grande_core.py:155-160` **Severity: 🟡 Medium**

Two permutes + advanced indexing, called in both `grande_forward()` and `build_grande_feature_stats()`. Consider `torch.gather` with pre-expanded indices or restructuring memory layout.

---

## 6. EXPERIMENT & LOGGING OVERHEAD

### Finding 6.1 — Profiling adds 8 `cuda.synchronize()` calls per forward pass
**File:** `ticl/models/decoders.py:1178-1214`, `ticl/models/mothernet.py:64-77` **Severity: 🟡 Medium (when `--grande-profile True`)**

Each profiled section calls `torch.cuda.synchronize()` at start and end — 8 pipeline stalls per batch.

**Fix:** Use CUDA events (non-blocking timing) or disable profiling for production runs.

### Finding 6.2 — Checkpoint saving and validation
**Severity: 🟢 Low**

Both run every `save_every` epochs (default 10). At ~4 min/epoch, blocking saves every ~40 minutes are acceptable.

---

## Ranked Action Plan

| Rank | Finding | File | Change | Impact |
|------|---------|------|--------|--------|
| **1** | 🔴 **Fix unused GradScaler** — real correctness bug on fp16 hardware | `train.py:104,108` | Add `scaler.scale/step/update` calls, or remove scaler if bf16-only | **Correctness fix** |
| **2** | 🟢 **`set_float32_matmul_precision('high')`** | `fit_model.py` | One line at startup | **TF32 for matmuls on Ampere+** |
| **3** | 🟢 **`optimizer.zero_grad(set_to_none=True)`** | `train.py:109` | One-word change | **~1-2% per step** |
| **4** | 🟢 **Allocate context tensors on GPU** + **pre-cast path_identifier_list** | `grande_core.py:131-135, 344` | Add `device=device`, cache dtype | **Eliminates 2 device transfers/batch** |
| **5** | 🟡 **`torch.compile(grande_forward, dynamic=True)`** | `grande_core.py:294` | Compile the kernel + fix `.item()` blocker | **~10-20% on forward kernel** |

**Lower priority:**
- **6.** Switch to `batch_first=True` for MHA fast path (~5-15% on attention)
- **7.** Full-layer gradient checkpointing if memory-bound
- **8.** Replace GRUCell with bf16-compatible state update (depthwise variant only)
- **9.** Vectorize ClassificationAdapter compatibility loop (requires batched unique-counting)

Items 1-4 are one-line changes. Item 5 requires testing compile compatibility.
