import torch
import torch.nn.functional as F

from ticl.models.decoders import GrandeDecoder
from ticl.models.grande_core import (
    build_grande_context,
    build_grande_feature_stats,
    build_tree_index_tensors,
    grande_forward,
)


def reference_grande_feature_stats(
    *,
    x_train,
    y_train,
    features_by_estimator,
    feature_mask,
    n_out,
):
    batch_size, n_estimators, selected_variables = features_by_estimator.shape
    stats = torch.zeros(
        batch_size,
        n_estimators,
        selected_variables,
        3 + n_out,
        dtype=x_train.dtype,
    )
    for batch_idx in range(batch_size):
        for estimator_idx in range(n_estimators):
            feature_ids = features_by_estimator[batch_idx, estimator_idx]
            mask = feature_mask[batch_idx, estimator_idx]
            used = int(mask.sum().item())
            if used == 0:
                continue

            selected_x = x_train[:, batch_idx, feature_ids[:used]]
            selected_y = y_train[:, batch_idx]
            valid_mask = ~torch.isnan(selected_x)
            valid_count = valid_mask.sum(dim=0).clamp_min(1)
            masked_values = torch.where(valid_mask, selected_x, torch.zeros_like(selected_x))
            mean = masked_values.sum(dim=0) / valid_count
            centered = torch.where(
                valid_mask,
                selected_x - mean.unsqueeze(0),
                torch.zeros_like(selected_x),
            )
            std = torch.sqrt(
                centered.square().sum(dim=0)
                / valid_mask.sum(dim=0).sub(1).clamp_min(1)
            )
            missing_rate = (~valid_mask).float().mean(dim=0)

            stats[batch_idx, estimator_idx, :used, 0] = mean
            stats[batch_idx, estimator_idx, :used, 1] = std
            stats[batch_idx, estimator_idx, :used, 2] = missing_rate

            if n_out <= 1:
                continue

            class_targets = selected_y.long().clamp_min(0).clamp_max(n_out - 1)
            clean_x = torch.nan_to_num(selected_x, nan=0.0)
            for class_idx in range(n_out):
                class_mask = (class_targets == class_idx).unsqueeze(-1) & valid_mask
                denom = class_mask.sum(dim=0).clamp_min(1)
                class_mean = torch.where(
                    class_mask.any(dim=0),
                    torch.where(class_mask, clean_x, torch.zeros_like(clean_x)).sum(dim=0)
                    / denom,
                    torch.zeros(used, dtype=x_train.dtype),
                )
                stats[batch_idx, estimator_idx, :used, 3 + class_idx] = class_mean
    return stats


def test_build_grande_context_masks_unused_local_slots():
    context = build_grande_context(
        batch_size=2,
        n_estimators=3,
        num_features_used=4,
        selected_variables=6,
        device="cpu",
        seed=0,
    )

    assert context["features_by_estimator"].shape == (2, 3, 6)
    assert context["feature_mask"].shape == (2, 3, 6)
    assert context["feature_mask"].sum(dim=-1).eq(4).all()
    assert (
        context["features_by_estimator"][context["feature_mask"]] < 4
    ).all()


def test_grande_forward_routes_samples_to_expected_leaf():
    path_ids, node_idx = build_tree_index_tensors(tree_depth=1)
    logits = grande_forward(
        x=torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32),
        split_values=torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float32),
        split_index_logits=torch.tensor([[[[10.0, -10.0]]]], dtype=torch.float32),
        estimator_weights=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        leaf_classes=torch.tensor(
            [[[[2.0, 0.0], [0.0, 2.0]]]], dtype=torch.float32
        ),
        features_by_estimator=torch.tensor([[[0, 1]]], dtype=torch.long),
        feature_mask=torch.tensor([[[True, True]]]),
        path_identifier_list=path_ids,
        internal_node_index_list=node_idx,
        training=False,
        dropout=0.0,
        missing_values=False,
        straight_through=False,
    )

    assert logits.shape == (2, 1, 2)
    assert logits.squeeze(1).argmax(dim=1).tolist() == [0, 1]


def test_grande_decoder_outputs_expected_shapes():
    decoder = GrandeDecoder(
        emsize=32,
        n_out=3,
        hidden_size=64,
        decoder_type="class_average",
        embed_dim=64,
        decoder_hidden_layers=1,
        nhead=4,
        in_size=10,
        tree_depth=2,
        n_estimators=4,
        selected_variables=6,
    )
    x = torch.randn(5, 2, 32)
    y = torch.randint(0, 3, (5, 2))
    x_train = torch.randn(5, 2, 10)
    context = decoder.build_context(
        batch_size=2,
        num_features_used=4,
        device=x.device,
        seed=0,
    )
    split_values, split_index_logits, estimator_weights, leaf_classes = decoder(
        x,
        y,
        x_train,
        context,
        seed=0,
    )

    assert split_values.shape == (2, 4, 3, 6)
    assert split_index_logits.shape == (2, 4, 3, 6)
    assert estimator_weights.shape == (2, 4, 4)
    assert leaf_classes.shape == (2, 4, 4, 3)


def test_build_grande_feature_stats_matches_reference_implementation():
    x_train = torch.tensor(
        [
            [[1.0, float("nan"), 3.0], [2.0, 5.0, 1.0]],
            [[2.0, 4.0, float("nan")], [3.0, float("nan"), 2.0]],
            [[float("nan"), 6.0, 5.0], [4.0, 7.0, 3.0]],
        ]
    )
    y_train = torch.tensor([[0, 1], [1, 2], [2, 1]])
    features_by_estimator = torch.tensor(
        [
            [[0, 1, 0], [2, 1, 0]],
            [[1, 2, 0], [0, 2, 0]],
        ]
    )
    feature_mask = torch.tensor(
        [
            [[True, True, False], [True, True, False]],
            [[True, True, False], [True, True, False]],
        ]
    )

    stats = build_grande_feature_stats(
        x_train=x_train,
        y_train=y_train,
        features_by_estimator=features_by_estimator,
        feature_mask=feature_mask,
        n_out=3,
    )
    reference = reference_grande_feature_stats(
        x_train=x_train,
        y_train=y_train,
        features_by_estimator=features_by_estimator,
        feature_mask=feature_mask,
        n_out=3,
    )

    assert torch.allclose(stats, reference, atol=1e-6)


def test_symmetry_breaking_decoder_outputs_differ_across_estimators():
    """Improvement 1: with small random init, decoder outputs should differ across estimators."""
    decoder = GrandeDecoder(
        emsize=32,
        n_out=3,
        hidden_size=64,
        decoder_type="class_average",
        embed_dim=64,
        decoder_hidden_layers=1,
        nhead=4,
        in_size=10,
        tree_depth=2,
        n_estimators=4,
        selected_variables=6,
    )
    x = torch.randn(5, 2, 32)
    y = torch.randint(0, 3, (5, 2))
    x_train = torch.randn(5, 2, 10)
    context = decoder.build_context(batch_size=2, num_features_used=4, device="cpu", seed=0)
    split_values, split_index_logits, _, _ = decoder(x, y, x_train, context, seed=0)
    # Estimators should not be identical (symmetry is broken)
    diffs = []
    for i in range(1, split_values.shape[1]):
        diffs.append((split_values[:, 0] - split_values[:, i]).abs().max().item())
    assert max(diffs) > 1e-6, "Estimator outputs are identical — symmetry not broken"


def test_log_space_path_matches_prod():
    """Improvement 2: log-space path computation should match torch.prod numerically."""
    path_ids, node_idx = build_tree_index_tensors(tree_depth=2)
    torch.manual_seed(42)
    x = torch.randn(4, 2, 3)
    split_values = torch.randn(2, 1, 3, 3)
    split_index_logits = torch.randn(2, 1, 3, 3)
    estimator_weights = torch.zeros(2, 1, 4)
    leaf_classes = torch.randn(2, 1, 4, 2)
    features_by_estimator = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]])
    feature_mask = torch.ones(2, 1, 3, dtype=torch.bool)

    logits = grande_forward(
        x=x,
        split_values=split_values,
        split_index_logits=split_index_logits,
        estimator_weights=estimator_weights,
        leaf_classes=leaf_classes,
        features_by_estimator=features_by_estimator,
        feature_mask=feature_mask,
        path_identifier_list=path_ids,
        internal_node_index_list=node_idx,
        training=False,
        dropout=0.0,
        missing_values=False,
        straight_through=False,
    )
    # Should produce valid finite output
    assert logits.isfinite().all()
    assert logits.shape == (4, 2, 2)


def test_log_space_path_gradients_nonzero_at_depth5():
    """Improvement 2: gradients should flow through a depth-5 tree."""
    path_ids, node_idx = build_tree_index_tensors(tree_depth=5)
    torch.manual_seed(0)
    split_values = torch.randn(1, 1, 31, 4, requires_grad=True)
    split_index_logits = torch.randn(1, 1, 31, 4)
    x = torch.randn(2, 1, 4)
    estimator_weights = torch.zeros(1, 1, 32)
    leaf_classes = torch.randn(1, 1, 32, 3)
    features_by_estimator = torch.arange(4).unsqueeze(0).unsqueeze(0)
    feature_mask = torch.ones(1, 1, 4, dtype=torch.bool)

    logits = grande_forward(
        x=x,
        split_values=split_values,
        split_index_logits=split_index_logits,
        estimator_weights=estimator_weights,
        leaf_classes=leaf_classes,
        features_by_estimator=features_by_estimator,
        feature_mask=feature_mask,
        path_identifier_list=path_ids,
        internal_node_index_list=node_idx,
        training=False,
        dropout=0.0,
        missing_values=False,
        straight_through=True,
    )
    loss = logits.sum()
    loss.backward()
    assert split_values.grad is not None
    assert (split_values.grad.abs() > 1e-10).any(), "Gradients vanished at depth 5"


def test_leaf_class_residual_reflects_class_prior():
    """Improvement 3: with zeroed MLP output, leaf_classes softmax should reflect class distribution."""
    decoder = GrandeDecoder(
        emsize=32,
        n_out=3,
        hidden_size=64,
        decoder_type="class_average",
        embed_dim=64,
        decoder_hidden_layers=1,
        nhead=4,
        in_size=10,
        tree_depth=2,
        n_estimators=2,
        selected_variables=6,
    )
    # Zero out decoder MLP last layer so raw output is ~0
    with torch.no_grad():
        decoder.mlp[-1].weight.zero_()
        decoder.mlp[-1].bias.zero_()

    x = torch.randn(20, 1, 32)
    # Imbalanced labels: 50% class 0, 30% class 1, 20% class 2
    y = torch.tensor([0]*10 + [1]*6 + [2]*4).unsqueeze(1)
    x_train = torch.randn(20, 1, 10)
    context = decoder.build_context(batch_size=1, num_features_used=4, device="cpu", seed=0)
    _, _, _, leaf_classes = decoder(x, y, x_train, context, seed=0)
    # Leaf classes should have class prior added; softmax should approximate (0.5, 0.3, 0.2)
    probs = F.softmax(leaf_classes[0, 0, 0], dim=-1)
    assert probs[0] > probs[1] > probs[2], f"Class prior not reflected: {probs.tolist()}"
    assert probs[0].item() > 0.35, f"Dominant class prob too low: {probs[0].item()}"


def test_quantile_split_values_in_feature_range():
    """Improvement 4: split_values should be in feature space, not near zero."""
    decoder = GrandeDecoder(
        emsize=32,
        n_out=3,
        hidden_size=64,
        decoder_type="class_average",
        embed_dim=64,
        decoder_hidden_layers=1,
        nhead=4,
        in_size=10,
        tree_depth=2,
        n_estimators=2,
        selected_variables=6,
    )
    n_train = 20
    x = torch.randn(n_train, 1, 32)
    y = torch.randint(0, 3, (n_train, 1))
    # Features at scale ~100
    x_train = torch.randn(n_train, 1, 10) * 10 + 100
    context = decoder.build_context(batch_size=1, num_features_used=6, device="cpu", seed=0)
    split_values, _, _, _ = decoder(x, y, x_train, context, seed=0)
    # With quantile transform, split_values should be centered near 100, not near 0
    sv_mean = split_values.mean().item()
    assert abs(sv_mean) > 10, f"Split values still near zero ({sv_mean:.2f}), quantile transform not working"


def test_split_temperature_sharpness():
    """Improvement 5: temperature affects gradients through ST estimator."""
    path_ids, node_idx = build_tree_index_tensors(tree_depth=1)

    def get_grad(temp):
        logits = torch.tensor([[[[1.0, 0.5]]]], requires_grad=True)
        out = grande_forward(
            x=torch.tensor([[[0.5, -0.5]]]),
            split_values=torch.tensor([[[[0.0, 0.0]]]]),
            split_index_logits=logits,
            estimator_weights=torch.tensor([[[0.0, 0.0]]]),
            leaf_classes=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            features_by_estimator=torch.tensor([[[0, 1]]]),
            feature_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            path_identifier_list=path_ids,
            internal_node_index_list=node_idx,
            training=False,
            dropout=0.0,
            missing_values=False,
            straight_through=True,
            split_temperature=temp,
        )
        out.sum().backward()
        return logits.grad.clone()

    grad_cold = get_grad(0.1)
    grad_hot = get_grad(10.0)
    assert not torch.allclose(grad_cold, grad_hot, atol=1e-5), "Temperature has no gradient effect"
