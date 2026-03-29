"""Random search HPO config sampler for factorized GRANDE MotherNet.

Samples random hyperparameter configurations, applies constraint filtering,
splits into chunks for parallel SLURM execution, and writes JSON outputs.

Usage:
    python ticl/configs/grande_hpo.py --n-trials 50 --seed 42 --n-jobs 10
"""

import argparse
import json
import math
import os
import time


SEARCH_SPACE = {
    "tree_depth": {"type": "categorical", "values": [3, 4, 5, 6]},
    "n_estimators": {"type": "categorical", "values": [16, 32, 64, 128]},
    "selected_variables": {"type": "categorical", "values": [4, 8, 16, 32]},
    "data_subset_fraction": {"type": "uniform", "low": 0.3, "high": 1.0},
    "grande_dropout": {"type": "categorical", "values": [0.0, 0.05, 0.1, 0.2, 0.3]},
    "grande_diversity_loss_weight": {"type": "categorical", "values": [0.0, 0.001, 0.005, 0.01]},
    "grande_output_init": {"type": "categorical", "values": ["default", "zero"]},
    "decoder_hidden_size": {"type": "categorical", "values": [1024, 2048, 4096]},
    "decoder_hidden_layers": {"type": "categorical", "values": [1, 2]},
    "decoder_embed_dim": {"type": "categorical", "values": [512, 1024]},
    "emsize": {"type": "categorical", "values": [256, 512, 1024]},
    "nlayers": {"type": "categorical", "values": [6, 12]},
    "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 3e-4},
    "weight_decay": {"type": "categorical", "values": [0.0, 0.01, 0.1]},
    "batch_size": {"type": "categorical", "values": [4, 8, 16]},
}

FIXED_VALUES = {
    "child_model": "grande",
    "grande_decoder_variant": "factorized_stats",
    "epochs": 50,
    "num_steps": 2048,
    "missing_values": True,
    "bootstrap": False,
    "validate": True,
    "save_every": 10,
    "progress_bar": False,
}


def sample_value(rng, spec):
    """Sample a single value from a search space spec."""
    if spec["type"] == "categorical":
        return rng.choice(spec["values"])
    elif spec["type"] == "uniform":
        return round(rng.uniform(spec["low"], spec["high"]), 4)
    elif spec["type"] == "log_uniform":
        log_low = math.log(spec["low"])
        log_high = math.log(spec["high"])
        return round(math.exp(rng.uniform(log_low, log_high)), 7)
    else:
        raise ValueError(f"Unknown spec type: {spec['type']}")


def sample_config(rng):
    """Sample one complete config from the search space."""
    return {name: sample_value(rng, spec) for name, spec in SEARCH_SPACE.items()}


def check_constraints(config):
    """Return True if config passes all constraints."""
    # OOM guard: 2^depth * n_estimators * batch_size <= 65536
    leaves = 2 ** config["tree_depth"]
    if leaves * config["n_estimators"] * config["batch_size"] > 65536:
        return False
    # Feature budget cap
    if config["selected_variables"] > 64:
        return False
    return True


def sample_configs(n_trials, seed):
    """Sample n_trials configs with rejection filtering."""
    import random
    rng = random.Random(seed)
    configs = []
    attempts = 0
    max_attempts = n_trials * 100
    while len(configs) < n_trials and attempts < max_attempts:
        cfg = sample_config(rng)
        attempts += 1
        if check_constraints(cfg):
            cfg["trial_id"] = len(configs)
            configs.append(cfg)
    if len(configs) < n_trials:
        print(f"Warning: only sampled {len(configs)}/{n_trials} configs after {max_attempts} attempts")
    return configs


# Map from config key to (CLI flag, argument group)
_FLAG_MAP = {
    # mothernet group
    "tree_depth": "--tree-depth",
    "n_estimators": "--n-estimators",
    "selected_variables": "--selected-variables",
    "data_subset_fraction": "--data-subset-fraction",
    "grande_dropout": "--grande-dropout",
    "grande_diversity_loss_weight": "--grande-diversity-loss-weight",
    "grande_output_init": "--grande-output-init",
    "decoder_hidden_size": "--decoder-hidden-size",
    "decoder_hidden_layers": "--decoder-hidden-layers",
    "decoder_embed_dim": "--decoder-embed-dim",
    # transformer backbone
    "emsize": "--emsize",
    "nlayers": "--nlayers",
    "child_model": "--child-model",
    "grande_decoder_variant": "--grande-decoder-variant",
    "missing_values": "--missing-values",
    "bootstrap": "--bootstrap",
    # optimizer group
    "learning_rate": "--learning-rate",
    "weight_decay": "--weight-decay",
    "epochs": "--epochs",
    # dataloader group
    "batch_size": "--batch-size",
    "num_steps": "--num-steps",
    # orchestration group
    "validate": "--validate",
    "save_every": "--save-every",
    "progress_bar": "--progress-bar",
}


def config_to_cli_flags(config):
    """Convert a config dict to a CLI flag string for fit_model.py."""
    merged = {**FIXED_VALUES, **config}
    parts = []
    for key, value in merged.items():
        if key == "trial_id":
            continue
        flag = _FLAG_MAP.get(key)
        if flag is None:
            raise KeyError(f"No CLI flag mapping for config key: {key}")
        parts.append(f"{flag} {value}")
    return " ".join(parts)


def split_into_chunks(trials, n_chunks=10):
    """Distribute trials across n_chunks as evenly as possible (round-robin)."""
    chunks = [[] for _ in range(n_chunks)]
    for i, trial in enumerate(trials):
        chunks[i % n_chunks].append(trial)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="GRANDE HPO random search config sampler")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of trials to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=10, help="Number of SLURM jobs (chunks)")
    args = parser.parse_args()

    sweep_id = f"grande_hpo_{args.seed}_{int(time.time())}"
    sweep_dir = os.path.join("runs", "hpo", sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)

    # Sample configs
    trials = sample_configs(args.n_trials, args.seed)
    print(f"Sampled {len(trials)} configs (seed={args.seed})")

    # Write all trials
    with open(os.path.join(sweep_dir, "trials.json"), "w") as f:
        json.dump(trials, f, indent=2)

    # Split into chunks and write each
    chunks = split_into_chunks(trials, args.n_jobs)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(sweep_dir, f"chunk_{i}.json"), "w") as f:
            json.dump(chunk, f, indent=2)
    print(f"Split into {args.n_jobs} chunks: {[len(c) for c in chunks]}")

    # Write search space for reproducibility
    with open(os.path.join(sweep_dir, "search_space.json"), "w") as f:
        json.dump(SEARCH_SPACE, f, indent=2)

    # Write sweep metadata
    meta = {
        "sweep_id": sweep_id,
        "seed": args.seed,
        "n_trials": len(trials),
        "n_jobs": args.n_jobs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixed_values": FIXED_VALUES,
    }
    with open(os.path.join(sweep_dir, "sweep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Sweep directory: {sweep_dir}")
    print(f"Sweep ID: {sweep_id}")


if __name__ == "__main__":
    main()
