import random
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import wandb

from ticl.models.grande_core import (
    flatten_grande_estimator_outputs,
    gather_estimator_features,
    pairwise_cosine_off_diag,
)
import ticl.utils as utils

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:  # pragma: no cover - sklearn is expected in normal environments
    f1_score = None
    roc_auc_score = None


DEFAULT_CONTEXT_SEEDS = (0, 1, 42)


def eval_criterion(criterion, targets, output, device, n_out):
    if isinstance(criterion, torch.nn.GaussianNLLLoss):
        assert output.shape[-1] == 2, (
            "need to write a little bit of code to handle multiple regression targets at once"
        )

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(
            mean_pred.flatten(),
            targets.to(device).flatten(),
            var=var_pred.flatten(),
        )
    elif isinstance(criterion, (torch.nn.MSELoss, torch.nn.BCEWithLogitsLoss)):
        losses = criterion(output.flatten(), targets.to(device).flatten())
    elif isinstance(criterion, torch.nn.CrossEntropyLoss):
        losses = criterion(
            output.reshape(-1, n_out)[:, : int(targets.max()) + 1],
            targets.to(device).long().flatten(),
        )
    else:
        losses = criterion(output, targets)
    losses = losses.view(*output.shape[0:2])
    return utils.torch_nanmean(losses.mean(0), return_nanshare=True)


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    return value


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def temporary_rng_seed(seed):
    if seed is None:
        yield
        return

    state = _capture_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        _restore_rng_state(state)


def prepare_grande_diagnostic_snapshot(dl, seed):
    with temporary_rng_seed(seed):
        batch = dl.get_test_batch()
    return _clone_to_cpu(batch)


@contextmanager
def temporary_recompute_attn_disabled(model):
    layers = getattr(getattr(model, "transformer_encoder", None), "layers", [])
    previous_values = []
    for layer in layers:
        if hasattr(layer, "recompute_attn"):
            previous_values.append((layer, layer.recompute_attn))
            layer.recompute_attn = False
    try:
        yield
    finally:
        for layer, previous_value in previous_values:
            layer.recompute_attn = previous_value


def _depth_node_indices(tree_depth, device):
    return [
        torch.arange(2**depth - 1, 2 ** (depth + 1) - 1, device=device, dtype=torch.long)
        for depth in range(tree_depth)
    ]


def _safe_scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().cpu().item())
    return float(value)


def _entropy(probs, dim=-1):
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=dim)


def _flatten_valid(values):
    if values is None:
        return None
    if torch.is_tensor(values):
        values = values.detach().float().cpu().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return None
        return values.numpy()
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return values


def _subsample(values, max_points):
    if values is None:
        return None
    if len(values) <= max_points:
        return values
    indices = np.linspace(0, len(values) - 1, num=max_points, dtype=np.int64)
    return values[indices]


def _make_histogram(values, max_points):
    flattened = _flatten_valid(values)
    if flattened is None:
        return None
    sampled = _subsample(flattened, max_points)
    if sampled is None or len(sampled) == 0:
        return None
    return wandb.Histogram(sampled)


def _pairwise_cosine_mean(vectors):
    off_diag = pairwise_cosine_off_diag(vectors)
    if off_diag is None:
        return 1.0
    return _safe_scalar(off_diag.mean())


def _pairwise_positive_cosine_mean(vectors):
    off_diag = pairwise_cosine_off_diag(vectors)
    if off_diag is None:
        return 0.0
    return _safe_scalar(off_diag.clamp_min(0).mean())


def _effective_dimension_95(vectors):
    effective_dims = []
    for batch_vectors in vectors:
        if batch_vectors.shape[0] <= 1:
            effective_dims.append(1.0)
            continue
        centered = batch_vectors - batch_vectors.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        variances = singular_values.square()
        total = variances.sum()
        if total <= 0:
            effective_dims.append(1.0)
            continue
        explained = torch.cumsum(variances / total, dim=0)
        effective_dims.append(float((explained < 0.95).sum().item() + 1))
    return float(np.mean(effective_dims)) if effective_dims else 1.0


def _ece(confidence, correct, n_bins=10):
    if confidence.numel() == 0:
        return 0.0
    bin_edges = torch.linspace(0, 1, steps=n_bins + 1, device=confidence.device)
    ece = torch.zeros((), device=confidence.device, dtype=torch.float32)
    total = confidence.numel()
    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            in_bin = (confidence >= left) & (confidence <= right)
        else:
            in_bin = (confidence >= left) & (confidence < right)
        if not in_bin.any():
            continue
        bin_confidence = confidence[in_bin].mean()
        bin_accuracy = correct[in_bin].float().mean()
        ece = ece + (in_bin.float().mean() * (bin_accuracy - bin_confidence).abs())
    return _safe_scalar(ece)


def _add_metric(metrics, key, value):
    if value is None:
        return
    metrics[key] = _safe_scalar(value)


def _collect_group_grad_stats(parameters):
    total_params = 0
    grad_sq_sum = 0.0
    grad_abs_sum = 0.0
    grad_max = 0.0
    weight_sq_sum = 0.0
    grad_elements = 0

    for parameter in parameters:
        if parameter is None or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_params += parameter.numel()
        grad_sq_sum += float(grad.float().pow(2).sum().item())
        grad_abs_sum += float(grad.float().abs().sum().item())
        grad_max = max(grad_max, float(grad.float().abs().max().item()))
        weight_sq_sum += float(parameter.detach().float().pow(2).sum().item())
        grad_elements += grad.numel()

    if total_params == 0 or grad_elements == 0:
        return None

    grad_norm = grad_sq_sum ** 0.5
    weight_norm = weight_sq_sum ** 0.5
    return {
        "l2_norm": grad_norm,
        "mean_abs": grad_abs_sum / grad_elements,
        "max_abs": grad_max,
        "grad_to_weight_ratio": grad_norm / max(weight_norm, 1e-12),
        "num_params": float(total_params),
    }


def _add_group_metrics(metrics, prefix, parameters):
    stats = _collect_group_grad_stats(parameters)
    if stats is None:
        return None
    for name, value in stats.items():
        metrics[f"{prefix}/{name}"] = value
    return stats


def _collect_parameter_grad_metrics(model, metrics):
    named_parameters = list(model.named_parameters())

    def parameters_with_prefix(prefix):
        return [parameter for name, parameter in named_parameters if name.startswith(prefix)]

    backbone_prefixes = ("encoder.", "y_encoder.", "transformer_encoder.")
    backbone_params = [
        parameter
        for name, parameter in named_parameters
        if name.startswith(backbone_prefixes)
    ]
    _add_group_metrics(metrics, "grande_diagnostics/gradients/params/backbone", backbone_params)
    _add_group_metrics(
        metrics,
        "grande_diagnostics/gradients/params/encoder",
        parameters_with_prefix("encoder."),
    )
    _add_group_metrics(
        metrics,
        "grande_diagnostics/gradients/params/y_encoder",
        parameters_with_prefix("y_encoder."),
    )
    _add_group_metrics(
        metrics,
        "grande_diagnostics/gradients/params/transformer_encoder",
        parameters_with_prefix("transformer_encoder."),
    )
    layer_norms = []
    for layer_idx in range(getattr(model.transformer_encoder, "num_layers", 0)):
        stats = _add_group_metrics(
            metrics,
            f"grande_diagnostics/gradients/params/transformer_layer_{layer_idx}",
            parameters_with_prefix(f"transformer_encoder.layers.{layer_idx}."),
        )
        if stats is not None:
            layer_norms.append(stats["l2_norm"])
    _add_group_metrics(
        metrics,
        "grande_diagnostics/gradients/params/decoder",
        parameters_with_prefix("decoder."),
    )

    decoder_subgroups = sorted(
        {
            name.split(".")[1]
            for name, _ in named_parameters
            if name.startswith("decoder.") and len(name.split(".")) > 2
        }
    )
    for subgroup in decoder_subgroups:
        _add_group_metrics(
            metrics,
            f"grande_diagnostics/gradients/params/decoder_{subgroup}",
            parameters_with_prefix(f"decoder.{subgroup}."),
        )
    return layer_norms


def _add_activation_grad_metrics(metrics, prefix, tensor):
    if tensor is None or tensor.grad is None:
        return None
    grad = tensor.grad.detach().float()
    stats = {
        "l2_norm": float(grad.norm().item()),
        "mean_abs": float(grad.abs().mean().item()),
        "max_abs": float(grad.abs().max().item()),
    }
    for name, value in stats.items():
        metrics[f"{prefix}/{name}"] = value
    return stats


def _classification_metrics(logits, targets, n_out):
    metrics = {}
    if n_out == 1:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        valid_mask = targets != -100
        logits = logits[valid_mask]
        targets = targets[valid_mask].float()
        if logits.numel() == 0:
            return metrics

        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).long()
        correct = predictions.eq(targets.long())
        confidence = torch.maximum(probs, 1.0 - probs)
        margins = (2.0 * confidence - 1.0).abs()

        metrics["grande_diagnostics/output/accuracy"] = float(correct.float().mean().item())
        metrics["grande_diagnostics/output/brier"] = float(((probs - targets) ** 2).mean().item())
        metrics["grande_diagnostics/output/ece"] = _ece(confidence, correct)
        metrics["grande_diagnostics/output/confidence_mean"] = float(confidence.mean().item())
        metrics["grande_diagnostics/output/margin_mean"] = float(margins.mean().item())
        metrics["grande_diagnostics/output/margin_std"] = float(margins.std(unbiased=False).item())
        metrics["grande_diagnostics/output/positive_prediction_rate"] = float(predictions.float().mean().item())
        if f1_score is not None:
            metrics["grande_diagnostics/output/f1"] = float(
                f1_score(
                    targets.cpu().numpy().astype(int),
                    predictions.cpu().numpy().astype(int),
                    zero_division=0,
                )
            )
        if roc_auc_score is not None:
            try:
                metrics["grande_diagnostics/output/roc_auc"] = float(
                    roc_auc_score(
                        targets.cpu().numpy().astype(int),
                        probs.cpu().numpy(),
                    )
                )
            except ValueError:
                pass
        return metrics

    logits = logits.reshape(-1, n_out)
    targets = targets.reshape(-1)
    valid_mask = targets != -100
    logits = logits[valid_mask]
    targets = targets[valid_mask].long()
    if logits.numel() == 0:
        return metrics

    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    correct = predictions.eq(targets)
    top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margins = top2[:, 0]
    else:
        margins = top2[:, 0] - top2[:, 1]
    confidence = probs.max(dim=-1).values
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=n_out).float()

    metrics["grande_diagnostics/output/accuracy"] = float(correct.float().mean().item())
    metrics["grande_diagnostics/output/brier"] = float(((probs - one_hot_targets) ** 2).sum(dim=-1).mean().item())
    metrics["grande_diagnostics/output/ece"] = _ece(confidence, correct)
    metrics["grande_diagnostics/output/confidence_mean"] = float(confidence.mean().item())
    metrics["grande_diagnostics/output/margin_mean"] = float(margins.mean().item())
    metrics["grande_diagnostics/output/margin_std"] = float(margins.std(unbiased=False).item())
    if f1_score is not None:
        metrics["grande_diagnostics/output/f1_macro"] = float(
            f1_score(
                targets.cpu().numpy(),
                predictions.cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        )
    if roc_auc_score is not None:
        try:
            metrics["grande_diagnostics/output/roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    targets.cpu().numpy(),
                    probs.cpu().numpy(),
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            pass
    return metrics


def _selected_feature_stats(debug):
    split_index_logits = debug["split_index_logits"].detach().float()
    feature_mask = debug["feature_mask"]
    features_by_estimator = debug["features_by_estimator"]

    masked_logits = split_index_logits.masked_fill(
        ~feature_mask.unsqueeze(2),
        torch.finfo(split_index_logits.dtype).min,
    )
    split_probs = torch.softmax(masked_logits, dim=-1)
    topk = split_probs.topk(k=min(2, split_probs.shape[-1]), dim=-1).values
    selected_local = masked_logits.argmax(dim=-1)
    selected_features = torch.gather(
        features_by_estimator.unsqueeze(2).expand(-1, -1, split_index_logits.shape[2], -1),
        dim=-1,
        index=selected_local.unsqueeze(-1),
    ).squeeze(-1)

    return masked_logits, split_probs, topk, selected_local, selected_features


def _append_histogram(metrics, key, values, level, hist_max_points):
    if level == "scalars":
        return
    histogram = _make_histogram(values, hist_max_points)
    if histogram is not None:
        metrics[key] = histogram


def _collect_structure_metrics(metrics, debug, data, level, hist_max_points):
    info, x, _ = data
    feature_stats = debug["feature_stats"].detach().float()
    split_values = debug["split_values"].detach().float()
    split_index_logits = debug["split_index_logits"].detach().float()
    leaf_classes = debug["leaf_classes"].detach().float()
    path_probs = debug["path_probs"].detach().float()
    estimator_weights_softmax = debug["estimator_weights_softmax"].detach().float()
    tree_depth = debug["tree_depth"]

    (
        masked_logits,
        split_probs,
        topk,
        selected_local,
        selected_features,
    ) = _selected_feature_stats(debug)
    depth_nodes = _depth_node_indices(tree_depth, split_index_logits.device)

    selected_thresholds = torch.gather(
        split_values,
        dim=-1,
        index=selected_local.unsqueeze(-1),
    ).squeeze(-1)
    selected_means = torch.gather(
        feature_stats[..., 0].unsqueeze(2).expand_as(split_values),
        dim=-1,
        index=selected_local.unsqueeze(-1),
    ).squeeze(-1)
    selected_stds = torch.gather(
        feature_stats[..., 1].unsqueeze(2).expand_as(split_values),
        dim=-1,
        index=selected_local.unsqueeze(-1),
    ).squeeze(-1).clamp_min(1e-6)
    threshold_z = (selected_thresholds - selected_means) / selected_stds

    # Feature selection metrics by depth.
    for depth, node_idx in enumerate(depth_nodes):
        depth_probs = split_probs[:, :, node_idx, :]
        depth_topk = topk[:, :, node_idx, :]
        depth_selected = selected_features[:, :, node_idx]
        entropy = _entropy(depth_probs, dim=-1)
        _add_metric(
            metrics,
            f"grande_diagnostics/feature_selection/depth_{depth}/entropy",
            entropy.mean(),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/feature_selection/depth_{depth}/mean_max_prob",
            depth_probs.max(dim=-1).values.mean(),
        )
        margin = depth_topk[..., 0]
        if depth_topk.shape[-1] > 1:
            margin = depth_topk[..., 0] - depth_topk[..., 1]
        _add_metric(
            metrics,
            f"grande_diagnostics/feature_selection/depth_{depth}/top1_top2_margin",
            margin.mean(),
        )
        unique_ratios = []
        concentration = []
        flattened_selected = depth_selected.reshape(depth_selected.shape[0], -1)
        for batch_selected in flattened_selected:
            if batch_selected.numel() == 0:
                continue
            unique_count = torch.unique(batch_selected).numel()
            unique_ratios.append(unique_count / batch_selected.numel())
            counts = torch.bincount(batch_selected, minlength=int(debug["num_features_used"]))
            probs = counts.float() / counts.sum().clamp_min(1)
            concentration.append(float(probs.square().sum().item()))
        if unique_ratios:
            metrics[
                f"grande_diagnostics/feature_selection/depth_{depth}/unique_ratio"
            ] = float(np.mean(unique_ratios))
        if concentration:
            metrics[
                f"grande_diagnostics/feature_selection/depth_{depth}/concentration_hhi"
            ] = float(np.mean(concentration))
        _append_histogram(
            metrics,
            f"grande_diagnostics/hists/selected_feature_depth_{depth}",
            depth_selected,
            level,
            hist_max_points,
        )

    # Threshold diagnostics on selected features only.
    for depth, node_idx in enumerate(depth_nodes):
        depth_thresholds = selected_thresholds[:, :, node_idx]
        depth_z = threshold_z[:, :, node_idx]
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/mean",
            depth_thresholds.mean(),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/std",
            depth_thresholds.std(unbiased=False),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/max_abs",
            depth_thresholds.abs().max(),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/zscore_abs_mean",
            depth_z.abs().mean(),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/variance",
            depth_thresholds.var(unbiased=False),
        )
        _add_metric(
            metrics,
            f"grande_diagnostics/thresholds/depth_{depth}/extreme_fraction_abs_z_gt_3",
            (depth_z.abs() > 3.0).float().mean(),
        )
        _append_histogram(
            metrics,
            f"grande_diagnostics/hists/threshold_zscore_depth_{depth}",
            depth_z,
            level,
            hist_max_points,
        )

    # Estimator diversity.
    split_vectors = split_index_logits.reshape(split_index_logits.shape[0], split_index_logits.shape[1], -1)
    threshold_vectors = selected_thresholds.reshape(selected_thresholds.shape[0], selected_thresholds.shape[1], -1)
    leaf_vectors = leaf_classes.reshape(leaf_classes.shape[0], leaf_classes.shape[1], -1)
    tree_vectors = torch.cat([split_vectors, threshold_vectors, leaf_vectors], dim=-1)
    combined_decoder_vectors = flatten_grande_estimator_outputs(
        split_values=split_values,
        split_index_logits=split_index_logits,
        estimator_weights=debug["estimator_weights"].detach().float(),
        leaf_classes=leaf_classes,
    )
    metrics["grande_diagnostics/diversity/split_index_cosine_mean"] = _pairwise_cosine_mean(split_vectors)
    metrics["grande_diagnostics/diversity/threshold_cosine_mean"] = _pairwise_cosine_mean(threshold_vectors)
    metrics["grande_diagnostics/diversity/leaf_cosine_mean"] = _pairwise_cosine_mean(leaf_vectors)
    metrics["grande_diagnostics/diversity/effective_dim_95"] = _effective_dimension_95(tree_vectors)
    metrics[
        "grande_diagnostics/diversity/combined_decoder_output_cosine_mean"
    ] = _pairwise_cosine_mean(combined_decoder_vectors)
    metrics[
        "grande_diagnostics/diversity/combined_decoder_output_positive_cosine_mean"
    ] = _pairwise_positive_cosine_mean(combined_decoder_vectors)
    combined_effective_dim = _effective_dimension_95(combined_decoder_vectors)
    metrics[
        "grande_diagnostics/diversity/combined_decoder_output_effective_dim_95"
    ] = combined_effective_dim
    metrics[
        "grande_diagnostics/diversity/combined_decoder_output_effective_dim_fraction"
    ] = combined_effective_dim / max(combined_decoder_vectors.shape[1], 1)

    # Estimator weighting and routing.
    estimator_entropy = _entropy(estimator_weights_softmax, dim=-1)
    effective_sizes = 1.0 / estimator_weights_softmax.square().sum(dim=-1).clamp_min(1e-12)
    metrics["grande_diagnostics/estimator_weights/entropy_mean"] = _safe_scalar(estimator_entropy.mean())
    metrics["grande_diagnostics/estimator_weights/effective_size_mean"] = _safe_scalar(effective_sizes.mean())
    metrics["grande_diagnostics/estimator_weights/max_weight_mean"] = _safe_scalar(
        estimator_weights_softmax.max(dim=-1).values.mean()
    )
    _append_histogram(
        metrics,
        "grande_diagnostics/hists/effective_estimator_size",
        effective_sizes,
        level,
        hist_max_points,
    )

    path_entropy = _entropy(path_probs, dim=-1)
    leaf_occupancy = path_probs.mean(dim=0)
    metrics["grande_diagnostics/routing/path_entropy_mean"] = _safe_scalar(path_entropy.mean())
    metrics["grande_diagnostics/routing/leaf_occupancy_mean"] = _safe_scalar(leaf_occupancy.mean())
    metrics["grande_diagnostics/routing/dead_leaf_fraction"] = _safe_scalar(
        (leaf_occupancy < 1e-4).float().mean()
    )
    _append_histogram(
        metrics,
        "grande_diagnostics/hists/leaf_occupancy",
        leaf_occupancy,
        level,
        hist_max_points,
    )

    x_test = x[debug["single_eval_pos"] :].detach()
    x_local = gather_estimator_features(x_test, debug["features_by_estimator"])
    selected_missing = torch.gather(
        torch.isnan(x_local),
        dim=-1,
        index=selected_local.unsqueeze(0).expand(x_local.shape[0], -1, -1, -1),
    )
    metrics["grande_diagnostics/routing/missing_value_routing_fraction"] = _safe_scalar(
        selected_missing.float().mean()
    )


def _collect_gradient_metrics(metrics, debug, hist_level, hist_max_points):
    layer_norms = []
    encoder_stats = _add_activation_grad_metrics(
        metrics,
        "grande_diagnostics/gradients/activations/encoder_output",
        debug.get("encoder_output"),
    )
    if encoder_stats is not None:
        layer_norms.append(encoder_stats["l2_norm"])
    for layer_idx, tensor in enumerate(debug.get("transformer_layer_outputs", [])):
        stats = _add_activation_grad_metrics(
            metrics,
            f"grande_diagnostics/gradients/activations/transformer_layer_{layer_idx}",
            tensor,
        )
        if stats is not None:
            layer_norms.append(stats["l2_norm"])
    transformer_output_stats = _add_activation_grad_metrics(
        metrics,
        "grande_diagnostics/gradients/activations/transformer_output",
        debug.get("transformer_output"),
    )
    if transformer_output_stats is not None:
        layer_norms.append(transformer_output_stats["l2_norm"])

    for tensor_name in (
        "split_values",
        "split_index_logits",
        "estimator_weights",
        "leaf_classes",
    ):
        _add_activation_grad_metrics(
            metrics,
            f"grande_diagnostics/gradients/activations/{tensor_name}",
            debug.get(tensor_name),
        )

    estimator_grad_vectors = {}
    for tensor_name in (
        "split_values",
        "split_index_logits",
        "estimator_weights",
        "leaf_classes",
    ):
        tensor = debug.get(tensor_name)
        if tensor is None or tensor.grad is None:
            continue
        grad_vectors = tensor.grad.detach().float().reshape(
            tensor.shape[0], tensor.shape[1], -1
        )
        estimator_grad_vectors[tensor_name] = grad_vectors
        metrics[
            f"grande_diagnostics/gradients/per_estimator/{tensor_name}_cosine_mean"
        ] = _pairwise_cosine_mean(grad_vectors)
        effective_dim = _effective_dimension_95(grad_vectors)
        metrics[
            f"grande_diagnostics/gradients/per_estimator/{tensor_name}_effective_dim_95"
        ] = effective_dim
        metrics[
            f"grande_diagnostics/gradients/per_estimator/{tensor_name}_effective_dim_fraction"
        ] = effective_dim / max(grad_vectors.shape[1], 1)

    if estimator_grad_vectors:
        combined_tree_output_grad = torch.cat(
            [
                estimator_grad_vectors["split_index_logits"],
                estimator_grad_vectors["split_values"],
                estimator_grad_vectors["estimator_weights"],
                estimator_grad_vectors["leaf_classes"],
            ],
            dim=-1,
        )
        metrics[
            "grande_diagnostics/gradients/per_estimator/combined_tree_output_cosine_mean"
        ] = _pairwise_cosine_mean(combined_tree_output_grad)
        combined_effective_dim = _effective_dimension_95(combined_tree_output_grad)
        metrics[
            "grande_diagnostics/gradients/per_estimator/combined_tree_output_effective_dim_95"
        ] = combined_effective_dim
        metrics[
            "grande_diagnostics/gradients/per_estimator/combined_tree_output_effective_dim_fraction"
        ] = combined_effective_dim / max(combined_tree_output_grad.shape[1], 1)

    split_values_grad = debug["split_values"].grad
    split_index_grad = debug["split_index_logits"].grad
    if split_values_grad is not None and split_index_grad is not None:
        for depth, node_idx in enumerate(_depth_node_indices(debug["tree_depth"], split_values_grad.device)):
            metrics[
                f"grande_diagnostics/gradients/per_depth/split_values_depth_{depth}_l2_norm"
            ] = float(split_values_grad[:, :, node_idx, :].detach().float().norm().item())
            metrics[
                f"grande_diagnostics/gradients/per_depth/split_index_logits_depth_{depth}_l2_norm"
            ] = float(split_index_grad[:, :, node_idx, :].detach().float().norm().item())
            metrics[
                f"grande_diagnostics/gradients/per_depth/split_values_depth_{depth}_cosine_mean"
            ] = _pairwise_cosine_mean(
                split_values_grad[:, :, node_idx, :].detach().float().reshape(
                    split_values_grad.shape[0],
                    split_values_grad.shape[1],
                    -1,
                )
            )
            metrics[
                f"grande_diagnostics/gradients/per_depth/split_index_logits_depth_{depth}_cosine_mean"
            ] = _pairwise_cosine_mean(
                split_index_grad[:, :, node_idx, :].detach().float().reshape(
                    split_index_grad.shape[0],
                    split_index_grad.shape[1],
                    -1,
                )
            )

    _append_histogram(
        metrics,
        "grande_diagnostics/hists/backbone_activation_grad_norms",
        np.asarray(layer_norms, dtype=np.float32) if layer_norms else None,
        hist_level,
        hist_max_points,
    )


def _collect_seed_stability_metrics(
    metrics,
    model,
    data,
    targets,
    single_eval_pos,
    criterion,
    device,
    n_out,
    base_seed,
    use_training_schedule,
):
    seeds = []
    for seed in (base_seed, *DEFAULT_CONTEXT_SEEDS):
        if seed not in seeds:
            seeds.append(seed)

    losses = []
    predictions = []
    probabilities = []
    with torch.no_grad():
        for seed in seeds:
            with temporary_rng_seed(seed):
                output = model(
                    data,
                    single_eval_pos=single_eval_pos,
                    grande_context_seed=seed,
                    advance_split_temperature=False,
                    grande_use_training_schedule=use_training_schedule,
                )
            eval_targets = targets[single_eval_pos:] if single_eval_pos is not None else targets
            loss, _ = eval_criterion(
                criterion,
                eval_targets,
                output,
                device=device,
                n_out=n_out,
            )
            losses.append(float(loss.detach().item()))
            if n_out == 1:
                probs = torch.sigmoid(output.detach().float()).reshape(-1)
                probabilities.append(probs)
                predictions.append((probs >= 0.5).long())
            else:
                probs = torch.softmax(output.detach().float(), dim=-1).reshape(-1, n_out)
                probabilities.append(probs)
                predictions.append(probs.argmax(dim=-1))

    metrics["grande_diagnostics/context_seed/loss_mean"] = float(np.mean(losses))
    metrics["grande_diagnostics/context_seed/loss_std"] = float(np.std(losses))
    metrics["grande_diagnostics/context_seed/loss_range"] = float(np.max(losses) - np.min(losses))

    if probabilities:
        stacked_probs = torch.stack(probabilities, dim=0).float()
        metrics["grande_diagnostics/context_seed/probability_variance_mean"] = float(
            stacked_probs.var(dim=0, unbiased=False).mean().item()
        )
    if len(predictions) > 1:
        agreements = []
        for left in range(len(predictions)):
            for right in range(left + 1, len(predictions)):
                agreements.append(
                    float(predictions[left].eq(predictions[right]).float().mean().item())
                )
        if agreements:
            metrics["grande_diagnostics/context_seed/prediction_agreement_mean"] = float(
                np.mean(agreements)
            )


def run_grande_diagnostics(
    *,
    model,
    snapshot,
    criterion,
    optimizer,
    device,
    n_out,
    epoch,
    level,
    hist_max_points,
    base_seed,
    collect_gradients=False,
):
    if getattr(model, "child_model", None) != "grande":
        return {}

    batch = _move_to_device(snapshot, device)
    data, targets, single_eval_pos = batch
    was_training = model.training
    metrics = {}

    try:
        model.eval()
        optimizer.zero_grad()
        with temporary_recompute_attn_disabled(model):
            with temporary_rng_seed(base_seed):
                grad_context = nullcontext() if collect_gradients else torch.no_grad()
                with grad_context:
                    output, debug = model(
                        data,
                        single_eval_pos=single_eval_pos,
                        return_debug=True,
                        grande_context_seed=base_seed,
                        advance_split_temperature=False,
                        grande_use_training_schedule=was_training,
                    )
                    eval_targets = targets[single_eval_pos:] if single_eval_pos is not None else targets
                    loss, nan_share = eval_criterion(
                        criterion,
                        eval_targets,
                        output,
                        device=device,
                        n_out=n_out,
                    )
                    loss = loss.mean()
                if collect_gradients:
                    loss.backward()

            metrics["grande_diagnostics/output/loss"] = float(loss.detach().item())
            metrics["grande_diagnostics/output/nan_share"] = float(nan_share)
            metrics["grande_diagnostics/meta/epoch"] = float(epoch)
            metrics["grande_diagnostics/meta/gradient_metrics_enabled"] = float(collect_gradients)
            metrics["grande_diagnostics/meta/split_temperature"] = float(
                model.decoder.peek_split_temperature(
                    device=device,
                    use_training_schedule=was_training,
                ).detach().cpu().item()
            )

            if collect_gradients:
                parameter_layer_norms = _collect_parameter_grad_metrics(model, metrics)
                _collect_gradient_metrics(metrics, debug, level, hist_max_points)
                _append_histogram(
                    metrics,
                    "grande_diagnostics/hists/backbone_param_grad_norms",
                    np.asarray(parameter_layer_norms, dtype=np.float32) if parameter_layer_norms else None,
                    level,
                    hist_max_points,
                )
            _collect_structure_metrics(metrics, debug, data, level, hist_max_points)
            metrics.update(
                _classification_metrics(
                    output.detach().float(),
                    eval_targets.detach(),
                    n_out=n_out,
                )
            )
            _collect_seed_stability_metrics(
                metrics,
                model,
                data,
                targets,
                single_eval_pos,
                criterion,
                device,
                n_out,
                base_seed,
                was_training,
            )
        return metrics
    finally:
        optimizer.zero_grad()
        if was_training:
            model.train()
        else:
            model.eval()
