# Plan: Integrate TRAINING_SPEED_AUDIT.md Optimizations

## Context

The TRAINING_SPEED_AUDIT.md identified 5 ranked optimization targets across the MotherNet/GRANDE training pipeline. The most critical is a correctness bug (GradScaler created but never used), followed by quick wins (TF32, zero_grad, GPU allocation) and an experimental torch.compile path. This plan implements them in dependency-aware order across 4 commits.

---

## Commit 1: Quick wins (Ranks 2, 3 + minor fix)

Single-line changes, zero risk.

### 1a. TF32 matmul precision — `ticl/fit_model.py`
- **Line 48**: After `torch.set_num_threads(24)`, add:
  ```python
  torch.set_float32_matmul_precision('high')
  ```
- Already present in standalone `grande.py:57`

### 1b. `set_to_none=True` — `ticl/train.py:109`
- Change `optimizer.zero_grad()` → `optimizer.zero_grad(set_to_none=True)`

### 1c. Remove redundant `.cpu()` — `ticl/train.py:114-115`
- Change `loss.mean().cpu().detach().item()` → `loss.mean().detach().item()`
- Same for `task_loss` on line 115 if applicable

---

## Commit 2: GPU tensor allocation & cast cleanup (Rank 4)

### 2a. Allocate on device — `ticl/models/grande_core.py:131-149`
- Lines 131-132: Add `device=device` to `torch.zeros(...)` for `features_by_estimator`
- Lines 134-135: Add `device=device` to `torch.zeros(...)` for `feature_mask`
- Lines 148-149: Remove `.to(device)` (already on device)
- **Verified**: `sampled_features` from `_sample_without_replacement` is already on `device` (line 79-84), so the assignment at line 144 works directly.

### 2b. Remove `.float()` cast — `ticl/models/grande_core.py:177`
- Change `torch.cat([part.float() for part in parts], dim=-1)` → `torch.cat(parts, dim=-1)`
- All parts come from the decoder at the same dtype. Only caller is diversity loss in `mothernet.py:159`.

### 2c. Pre-cast `path_identifier_list` — avoid per-forward `.to(dtype)` at `grande_core.py:344`

**Approach**: Register a non-persistent float buffer alongside the existing long buffer, avoiding checkpoint compatibility issues.

Files to change:
1. **`ticl/models/decoders.py:905-906`** — After existing buffer registration, add:
   ```python
   self.register_buffer("path_identifier_list_float", path_identifier_list.float(), persistent=False)
   ```
2. **`ticl/models/mothernet.py`** — At the `grande_forward(...)` call (~line 346), pass `self.decoder.path_identifier_list_float` instead of `self.decoder.path_identifier_list`
3. **`ticl/models/grande_core.py:344`** — Remove the `.to(dtype=dtype)` cast, just use `path_identifier_list` directly (it arrives as float32, autocast handles the rest)
4. **`ticl/prediction/mothernet.py:376`** — Extraction: change to `detach(model.decoder.path_identifier_list_float)`
5. **`ticl/prediction/mothernet.py:691-692`** — Inference: change `dtype=torch.long` → `dtype=torch.float32` for `path_identifier_list`

**Note**: `internal_node_index_list` stays as `torch.long` — it's used as an index tensor, not in arithmetic.

---

## Commit 3: Fix unused GradScaler (Rank 1)

**File**: `ticl/train.py`

### Backward pass (line 104)
```python
# BEFORE
loss.backward()

# AFTER
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()
```

### Optimizer step block (lines 106-109)
```python
# BEFORE
if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
    optimizer.step()
    optimizer.zero_grad()

# AFTER
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
```

**Key details**:
- `scaler.unscale_()` must come before `clip_grad_norm_` so clipping sees true-scale gradients
- `scaler.step()` internally skips `optimizer.step()` if inf/nan gradients are detected
- Gradient accumulation is correct: `scaler.scale(loss).backward()` every batch, `unscale_/step/update` only on aggregation boundary
- On bf16 GPUs this is harmless (no underflow to prevent); on fp16 GPUs this fixes silent gradient underflow

---

## Commit 4: Fix `.item()` sync + optional `torch.compile` (Rank 5)

### 4a. Fix `.item()` blocker — `ticl/models/decoders.py:942-952`
```python
# BEFORE
progress = min(
    float(self.split_temperature_step.item())
    / float(self.grande_split_temperature_anneal_steps),
    1.0,
)
temperature = self.grande_split_temperature_start + progress * (
    self.grande_split_temperature_end - self.grande_split_temperature_start
)
...
return torch.tensor(temperature, device=device, dtype=torch.float32)

# AFTER
progress = torch.clamp(
    self.split_temperature_step.float()
    / float(self.grande_split_temperature_anneal_steps),
    max=1.0,
)
temperature = self.grande_split_temperature_start + progress * (
    self.grande_split_temperature_end - self.grande_split_temperature_start
)
...
return temperature.to(device=device, dtype=torch.float32)
```
Independently valuable — eliminates a CUDA sync per forward pass with depthwise decoder.

### 4b. `torch.compile` behind `--grande-compile` flag

Files:
1. **`ticl/cli_parsing.py`** — Add `--grande-compile` boolean arg (default `False`)
2. **`ticl/model_configs.py`** — Add `"grande_compile": False` to GRANDE config section
3. **`ticl/models/mothernet.py`** — In `MotherNet.__init__`, conditionally compile:
   ```python
   if grande_compile:
       self._compiled_grande_forward = torch.compile(grande_forward, dynamic=True)
   ```
   In `ModelPredictor.forward`, use compiled version when available:
   ```python
   forward_fn = getattr(self, '_compiled_grande_forward', grande_forward)
   ```

Uses `dynamic=True` because `single_eval_pos` varies per batch (no static shapes for CUDA graphs).

---

## Critical files
| File | Changes |
|------|---------|
| `ticl/train.py` | GradScaler fix, zero_grad, .cpu() removal |
| `ticl/models/grande_core.py` | GPU alloc, remove .float() cast, remove path_id .to() |
| `ticl/models/decoders.py` | Float buffer registration, .item() fix |
| `ticl/fit_model.py` | TF32 matmul precision |
| `ticl/models/mothernet.py` | Pass float buffer, torch.compile |
| `ticl/prediction/mothernet.py` | Sync path_identifier_list dtype in extract + predict |
| `ticl/cli_parsing.py` | --grande-compile flag |
| `ticl/model_configs.py` | grande_compile default |

## Verification

After each commit:
1. `pytest ticl/tests/ -x` — all existing tests must pass
2. After Commit 2: `pytest ticl/tests/test_grande_core.py -v`

GPU validation (manual, before merging):
```bash
python ticl/fit_model.py mothernet \
    --child-model grande --tree-depth 3 --n-estimators 16 \
    --selected-variables 10 --num-steps 128 --epochs 5 \
    --grande-profile True
```
Compare `train_gpu_time` and per-component timings before/after.
