import random

import numpy as np
import torch
import wandb
from torch import nn

from ticl.grande_diagnostics import (
    prepare_grande_diagnostic_snapshot,
    run_grande_diagnostics,
)
from ticl.models.mothernet import MotherNet
from ticl.train import train


def _clone_batch(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    return value


def _make_grande_batch():
    torch.manual_seed(0)
    n_samples = 6
    batch_size = 2
    n_features = 5
    x = torch.randn(n_samples, batch_size, n_features)
    x[1, 0, 2] = float("nan")
    y = torch.randint(0, 3, (n_samples, batch_size))
    info = {"num_features_used": n_features}
    single_eval_pos = 4
    return ((info, x, y), y.clone(), single_eval_pos)


def _make_grande_model(*, diagnostics=False):
    return MotherNet(
        n_out=3,
        emsize=16,
        nhead=4,
        nhid_factor=2,
        nlayers=2,
        n_features=5,
        child_model="grande",
        tree_depth=2,
        n_estimators=2,
        dropout=0.0,
        y_encoder_layer=None,
        decoder_type="class_average",
        decoder_embed_dim=32,
        classification_task=True,
        decoder_hidden_layers=1,
        decoder_hidden_size=32,
        decoder_activation="relu",
        selected_variables=4,
        grande_decoder_variant="factorized_stats",
        grande_diagnostics=diagnostics,
        grande_diagnostics_level="scalars_small_hists",
        grande_diagnostics_seed=7,
        grande_diagnostics_hist_max_points=128,
    )


class _RandomBatchLoader:
    def get_test_batch(self):
        data = torch.tensor(
            [
                random.random(),
                np.random.rand(),
                torch.rand(1).item(),
            ],
            dtype=torch.float32,
        )
        x = data.view(1, 1, -1)
        y = torch.zeros(1, 1, dtype=torch.long)
        return (({"num_features_used": 3}, x, y), y.clone(), 0)


class _SingleBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.model = None

    def __len__(self):
        return 1

    def __iter__(self):
        yield _clone_batch(self.batch)

    def get_test_batch(self):
        return _clone_batch(self.batch)


def test_prepare_grande_diagnostic_snapshot_is_deterministic_and_restores_rng_state():
    loader = _RandomBatchLoader()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    snapshot_a = prepare_grande_diagnostic_snapshot(loader, seed=7)
    after_python = random.random()
    after_numpy = np.random.rand()
    after_torch = torch.rand(1)
    snapshot_b = prepare_grande_diagnostic_snapshot(loader, seed=7)

    assert torch.equal(snapshot_a[0][1], snapshot_b[0][1])
    assert after_python == expected_python
    assert after_numpy == expected_numpy
    assert torch.allclose(after_torch, expected_torch)


def test_run_grande_diagnostics_collects_backbone_and_grande_gradients():
    model = _make_grande_model(diagnostics=True)
    snapshot = _make_grande_batch()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_grande_diagnostics(
        model=model,
        snapshot=snapshot,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        n_out=3,
        epoch=0,
        level="scalars_small_hists",
        hist_max_points=128,
        base_seed=7,
    )

    assert metrics["grande_diagnostics/gradients/params/backbone/l2_norm"] > 0
    assert metrics["grande_diagnostics/gradients/params/decoder/l2_norm"] > 0
    assert metrics["grande_diagnostics/gradients/activations/transformer_output/l2_norm"] > 0
    assert metrics["grande_diagnostics/gradients/activations/split_index_logits/l2_norm"] > 0
    assert "grande_diagnostics/thresholds/depth_0/zscore_abs_mean" in metrics
    assert "grande_diagnostics/diversity/effective_dim_95" in metrics
    assert all(
        parameter.grad is None or torch.allclose(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in model.parameters()
    )


def test_train_logs_grande_diagnostics_at_epoch_zero_and_each_epoch(monkeypatch):
    logs = []

    def fake_log(payload, step=None):
        logs.append((payload, step))

    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", fake_log)

    model = _make_grande_model(diagnostics=True)
    loader = _SingleBatchLoader(_make_grande_batch())
    criterion = nn.CrossEntropyLoss(reduction="none")

    train(
        loader,
        model,
        criterion=criterion,
        epochs=2,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_epochs=1,
        device="cpu",
        progress_bar=False,
        verbose=False,
    )

    diagnostic_steps = [
        step
        for payload, step in logs
        if "grande_diagnostics/output/loss" in payload
    ]

    assert diagnostic_steps == [0, 1, 2]
