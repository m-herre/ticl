# autoresearch-ticl

Autonomous research loop for optimizing MotherNet-GRANDE — a hypernetwork that predicts differentiable decision tree ensemble parameters.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch-ticl/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-ticl/<tag>` from current branch.
3. **Read the in-scope files**: Read these files for full context:
   - `CLAUDE.md` — project architecture, conventions, and the three-code-path rule.
   - `autoresearch_ticl/program.md` — this file (experiment rules).
   - `autoresearch_ticl/config.py` — editable hyperparameters.
   - `autoresearch_ticl/run_experiment.py` — experiment runner (read-only).
   - `ticl/models/grande_core.py` — GRANDE forward kernel, context sampling, feature stats.
   - `ticl/models/decoders.py` — GrandeDecoder class.
   - `ticl/models/mothernet.py` — ModelPredictor.forward() GRANDE branch.
   - `ticl/prediction/mothernet.py` — extract_grande_model(), predict_with_grande_model().
4. **Verify baseline checkpoint exists**: Check that the file in `config.py:WARM_START_FROM` exists.
5. **Initialize results.tsv**: Create `autoresearch_ticl/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. Training runs for a fixed budget of `EPOCHS` epochs (default 5) with `NUM_STEPS` steps per epoch (default 256), followed by a validation pass on OpenML datasets.

You launch it simply as: `python autoresearch_ticl/run_experiment.py > autoresearch_ticl/run.log 2>&1`

Or, if you want to see the summary in your context while still logging:

```bash
python autoresearch_ticl/run_experiment.py 2>&1 | tee autoresearch_ticl/run.log
```

**What you CAN edit:**

- **Tier 1 — Hyperparameters (always safe):**
  - `autoresearch_ticl/config.py` — all constants except the "fixed" section (NUM_STEPS, EPOCHS, SAVE_EVERY).

- **Tier 2 — GRANDE model code (be careful, follow the three-code-path rule):**
  - `ticl/models/grande_core.py` — GRANDE forward kernel, context sampling, feature stats
  - `ticl/models/decoders.py` — GrandeDecoder class
  - `ticl/models/mothernet.py` — ModelPredictor.forward() GRANDE branch
  - `ticl/prediction/mothernet.py` — extract_grande_model(), predict_with_grande_model()

**What you CANNOT edit:**

- `autoresearch_ticl/run_experiment.py` — runner infrastructure (read-only)
- `ticl/train.py` — training loop
- `ticl/dataloader.py` — data loading
- `ticl/fit_model.py` — CLI entry point
- `ticl/cli_parsing.py` — argument parsing
- `ticl/utils.py` — utilities and validation callback
- `ticl/evaluation/*` — evaluation harness
- `grande.py` — reference GRANDE implementation

### ⚠️  THREE CODE PATH RULE ⚠️

Changes to decoder output shapes in `decoders.py` (GrandeDecoder) **must** be propagated to all three code paths:

1. **Training forward**: `ticl/models/mothernet.py` → `ModelPredictor.forward()` grande branch
2. **Parameter extraction**: `ticl/prediction/mothernet.py` → `extract_grande_model()`
3. **Standalone inference**: `ticl/prediction/mothernet.py` → `predict_with_grande_model()`

The `grande_forward()` kernel in `ticl/models/grande_core.py` is shared by all three — changes there propagate automatically. But if you change what the decoder *outputs* or how those outputs are *named/shaped*, you must update all consumers.

**The goal is simple: get the highest val_score.** This is the mean ROC-AUC across OpenML validation datasets. Higher is better.

**Minimum improvement threshold**: 0.005. Improvements below this threshold are not worth keeping unless they also simplify the code.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run the experiment with the default config.py as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_score:        0.7504
train_loss:       0.6120
total_seconds:    580.2
exit_code:        0
```

You can extract the key metric from the log file:

```bash
grep "^val_score:" autoresearch_ticl/run.log
```

## Logging results

When an experiment is done, log it to `autoresearch_ticl/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_score	train_loss	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.7504) — use 0.000000 for crashes
3. train_loss (e.g. 0.6120) — use 0.0000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_score	train_loss	status	description
a1b2c3d	0.7504	0.6120	keep	baseline
b2c3d4e	0.7560	0.5980	keep	increase n_estimators to 128
c3d4e5f	0.7490	0.6200	discard	switch decoder_type to output_attention
d4e5f6g	0.0000	0.0000	crash	double emsize (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-ticl/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Plan your experiment: review past results in results.tsv, identify what to try next.
3. Make changes to `autoresearch_ticl/config.py` and/or the Tier 2 model files.
4. `git add` changed files and commit with a descriptive message.
5. Run the experiment: `python autoresearch_ticl/run_experiment.py > autoresearch_ticl/run.log 2>&1`
6. Read out the results: `grep "^val_score:\|^train_loss:" autoresearch_ticl/run.log`
7. If the grep output is empty or shows "CRASH", the run crashed. Run `tail -n 50 autoresearch_ticl/run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on that idea.
8. Record the results in results.tsv (NOTE: do not commit results.tsv — leave it untracked by git).
9. If val_score improved by ≥ 0.005 (higher is better), you "advance" the branch, keeping the git commit.
10. If val_score did not improve by ≥ 0.005, `git reset --hard HEAD~1` to revert.

**Timeout**: Each experiment should take ~10 minutes total (training + validation). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope model files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are some promising directions (not exhaustive — be creative):

**Hyperparameter tuning (Tier 1, config.py only):**
- Adjust tree depth (3, 4, 5, 6)
- Vary n_estimators (32, 64, 128, 256)
- Tune selected_variables
- Experiment with learning rate (1e-5 to 1e-4)
- Try weight decay (1e-4, 1e-3)
- Adjust decoder_embed_dim
- Enable bootstrap / data_subset_fraction
- Enable grande_dropout (0.1, 0.2, 0.5)

**Architecture changes (Tier 2, model code):**
- Improve estimator weight aggregation (instance-dependent vs uniform)
- Better feature selection mechanisms
- Improved split threshold parameterization
- Nan-aware routing enhancements
- Decoder architecture improvements
- Gradient flow improvements (initialization, normalization)
- Regularization techniques specific to tree ensembles
