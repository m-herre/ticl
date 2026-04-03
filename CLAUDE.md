# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation: FILES.md Index

Every directory contains a `FILES.md` describing each file. When starting a task:

1. **Read the relevant `FILES.md` files first** before opening source files.
2. When you create, rename, or delete files, **update the corresponding `FILES.md`**.

## Research Methodology

This is a **research project**. We follow an evidence-based approach:

1. **Hypothesize** — Develop a concrete hypothesis about what might improve performance or explain a behavior.
2. **Design test** — Decide how to test the hypothesis (experiment, ablation, diagnostic, etc.).
3. **Execute & measure** — Run the experiment, collect results.
4. **Record** — Log findings in `EXPERIMENTS.md` so we build cumulative knowledge.

Do **not** jump straight to fixing or changing code. Always start with a hypothesis and a plan to validate it.

## Project Goal

Build a **MotherNet for GRANDE** — a transformer backbone that predicts the parameters of a **hard, axis-aligned, differentiable decision tree ensemble** (like GRANDE) instead of MLP weights. The end goal is an in-context learner that outputs interpretable tree ensembles with crisp, axis-aligned splits.

**Current status**: The GRANDE MotherNet is functional but does **not yet match MLP MotherNet performance**. The primary research goal is to close this gap — at minimum matching MLP MotherNet accuracy.

Key files:

- `grande.py` — standalone GRANDE reference implementation
- `ticl/models/mothernet.py` — MotherNet model (`ModelPredictor`, `MotherNet`)
- `ticl/models/decoders.py` — `GrandeDecoder` (and legacy `GradTreeDecoder`, `MLPModelDecoder`)
- `ticl/models/grande_core.py` — shared GRANDE context sampling, feature statistics, and tree forward kernel

## Architecture Overview

```
Training data (x, y) -> Encoder -> TransformerEncoder -> GrandeDecoder -> GRANDE params -> grande_forward(x_test)
```

1. **Encoder**: Linear maps input features to `emsize`-dim embeddings. Y is encoded and added.
2. **Transformer backbone**: `TransformerEncoderSimple` (batch_first=False). Produces per-sample contextual embeddings.
3. **GrandeDecoder**: Conditioned on transformer output + labels, produces child model parameters: `split_values`, `split_index_logits`, `estimator_weights`, `leaf_classes`.
4. **grande_forward()**: Shared kernel for training and inference. Differentiable tree ensemble with straight-through estimators.

### GrandeDecoder Variants (`--grande-decoder-variant`)

- **baseline**: Single MLP outputs all params at once
- **factorized_stats**: Four separate MLPs for split_values, split_index, estimator_weights, leaf_classes
- **depthwise_factorized_stats**: GRU-based recurrent decoder that processes tree levels sequentially

### grande_forward() Kernel (ticl/models/grande_core.py)

1. Feature selection: softmax + ST hard one-hot on split_index_logits
2. Split decision: softsign on (threshold - selected_feature_value), then ST round
3. Path probability: product over depth using precomputed path indices
4. Optional NaN-aware routing (missing values sent down both branches)
5. Instance-dependent weighted ensemble aggregation (softmax over estimator_weights)
6. Optional dropout on estimator weights during training

### GRANDE Context (ticl/models/grande_core.py)

- `build_grande_context()`: Samples feature subsets per estimator (`features_by_estimator`)
- `build_grande_feature_stats()`: Per-estimator statistics (mean, std, missing rate, class-conditional means)
- Supports data subsetting and bootstrap for estimator-local statistics

## Commands

```bash
# Install
conda create -f environment.yml && conda activate ticl && pip install -e .

# Train MotherNet (GRANDE child)
python ticl/fit_model.py mothernet \
    --child-model grande \
    --tree-depth 4 \
    --n-estimators 64 \
    --selected-variables 16 \
    --num-steps 2048 \
    --use-wandb

# Train MotherNet (MLP child, legacy default)
python ticl/fit_model.py mothernet

# Continue from checkpoint
python ticl/fit_model.py mothernet --continue-run --warm-start-from <path.cpkt>

# See all options
python ticl/fit_model.py mothernet -h
```

### SLURM Batch Jobs

```bash
# Submit a batch job
sbatch continue_gradtree.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View logs
tail -f logs/gradtree_<job_id>.out
```

Batch scripts use `#SBATCH` directives. See `continue_gradtree.sh` for the current GRANDE training template (20 CPUs, 1 GPU, 50GB RAM, partition `gpu-vram-94gb`).

### Interactive GPU Sessions

```bash
# Get an interactive GPU session
salloc --partition=gpu-vram-94gb --gres=gpu:1 --cpus-per-task=12 --mem=50G

# Then run training interactively
python ticl/fit_model.py mothernet --child-model grande ...
```

## GPU Usage and H100 Tuning

### What works by default

- Single-GPU CUDA training is the default path. If `--use-cpu` is not set, training uses CUDA automatically.
- Mixed precision is enabled by default. On H100, that means bf16 autocast by default.
- `torch.set_float32_matmul_precision("high")` is enabled at startup, so TF32 matmuls are already on.
- Flash SDP is left enabled on SM90 GPUs such as H100.

### What is not optimized by default

- The default physical batch size is conservative (`--batch-size 8`).
- Attention recomputation is enabled by default (`--recompute-attn True`), which saves memory but reduces throughput.
- `--grande-compile` is off by default.
- Do not assume all visible GPUs will be used automatically. The normal local multi-GPU launch path is not a supported default workflow; prefer single-GPU runs unless you are explicitly working on distributed training.

### H100 Throughput Preset

Use this when you want maximum single-run throughput on a single H100:

```bash
python ticl/fit_model.py mothernet \
    -g 0 \
    -b 64 \
    -k 1 \
    -t true \
    --recompute-attn false \
    --adaptive-batch-size false \
    --progress-bar false \
    --save-every 100
```

For GRANDE runs, add:

```bash
    --child-model grande \
    --grande-compile true
```

Tuning guidance:

- First increase `--batch-size` until you are close to memory limit.
- Keep `--aggregate_k_gradients 1` while tuning for raw throughput. Gradient accumulation increases effective batch size, but it does not improve GPU utilization the way a larger physical batch does.
- Only keep `--recompute-attn true` if you need it to fit memory.
- `--grande-compile true` is most useful for longer GRANDE runs where compile overhead can amortize.

### Fast Prototyping Preset

Use this when iteration speed matters more than maximizing tokens/sec over a long run:

```bash
python ticl/fit_model.py mothernet \
    -g 0 \
    -b 32 \
    -k 1 \
    -t true \
    --recompute-attn false \
    --adaptive-batch-size false \
    --num-steps 512 \
    -E 20 \
    --save-every 100 \
    --validate false \
    --progress-bar false
```

For very short smoke tests, leave `--grande-compile` off. For medium or long GRANDE experiments, turn it on.

### Benchmarking Guidance

- For apples-to-apples benchmarking, set `--seed-everything true`.
- Disable diagnostics and profiling unless they are the subject of the experiment.
- Keep `--grande-profile false` and `--grande-diagnostics false` for throughput-focused runs.
- `--detect-anomaly` should remain off except for debugging.
- Logging is useful, but W&B / MLflow / validation all add overhead. For pure throughput measurement, keep the run minimal.

## Tensor Conventions

- Transformer operates **batch_first=False**: sequences are `(seq_len, batch, emsize)`
- `x` input: `(n_samples, batch, n_features)`, split at `single_eval_pos` into train/test
- `info` dict may contain `num_features_used` for masking padded features
- Decoders output per-batch parameters; test inference is vectorized over `n_test` samples

## GRANDE Code Paths — Keep In Sync

All three paths use the **same `grande_forward()` kernel**. Any change to decoder output shapes must be propagated to all three:

1. **Training forward**: `ticl/models/mothernet.py` -> `ModelPredictor.forward()` (grande branch)
2. **Parameter extraction**: `ticl/prediction/mothernet.py` -> `extract_grande_model()`
3. **Standalone inference**: `ticl/prediction/mothernet.py` -> `predict_with_grande_model()`

Key invariants:
- `GrandeDecoder` does **not** predict `features_by_estimator`. Those are sampled externally via `build_grande_context()` and must be carried through extraction/inference.
- Row subsampling / bootstrap affect estimator-local feature statistics, not the inference-time kernel directly.
- Avoid reintroducing separate numpy/CUDA implementations — the shared kernel is intentional.

## Training Time

Full MotherNet training is slow: ~4 min/epoch at `--num-steps 2048`, ~12 min/epoch at `--num-steps 8192`. Keep this in mind when testing or experimenting — prefer small-scale runs (fewer steps/estimators/depth) for quick iteration, and reserve full-scale runs for validation.

## Performance Notes

- On H100, the main practical speed levers are larger physical batch size and disabling attention recomputation when memory allows.
- Mixed precision, TF32 matmuls, and flash SDP are already enabled by default on H100-capable runs.
- Use `--grande-compile true` for longer GRANDE experiments; skip it for very short prototyping runs.
- Runtime hot spots are in `ticl/models/grande_core.py`, especially context sampling and estimator-local feature statistics. Keep those paths vectorized; avoid Python loops over `batch_size * n_estimators`.
- Use `--grande-profile True` to record epoch-level timings: `grande_context_s`, `grande_feature_stats_s`, `grande_decoder_mlp_s`, `grande_forward_s`.

## Legacy: GradTree Path

The `child_model="gradtree"` path is **legacy**. It predicts `(I_logits, T, L)` and uses a separate `tree_forward()` in `mothernet.py`. The GRANDE path supersedes it. Do not invest in GradTree unless specifically asked.
