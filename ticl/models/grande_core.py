import math

import numpy as np
import torch
import torch.nn.functional as F


def one_hot_argmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    idx = logits.argmax(dim=dim, keepdim=True)
    return torch.zeros_like(logits).scatter_(dim, idx, 1.0)


def st(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return soft - (soft - hard).detach()


def resolve_selected_variables(selected_variables, max_features):
    if selected_variables <= 0:
        raise ValueError("selected_variables must be positive")
    if selected_variables <= 1:
        resolved = int(max_features * selected_variables)
        resolved = min(resolved, 50)
        resolved = max(resolved, 10)
        resolved = min(resolved, max_features)
    else:
        resolved = min(int(round(selected_variables)), max_features)
    return max(resolved, 1)


def build_tree_index_tensors(tree_depth):
    n_leaves = 2**tree_depth
    path_identifier_list = []
    internal_node_index_list = []
    for leaf_index in range(n_leaves):
        for current_depth in range(1, tree_depth + 1):
            path_identifier = (
                leaf_index // (2 ** (tree_depth - current_depth))
            ) % 2
            internal_node_index = (
                (2 ** (current_depth - 1))
                + (leaf_index // 2 ** (tree_depth - (current_depth - 1)))
                - 1
            )
            path_identifier_list.append(path_identifier)
            internal_node_index_list.append(internal_node_index)
    return (
        torch.tensor(
            np.reshape(np.array(path_identifier_list), (-1, tree_depth)),
            dtype=torch.long,
        ),
        torch.tensor(
            np.reshape(np.array(internal_node_index_list), (-1, tree_depth)),
            dtype=torch.long,
        ),
    )


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _make_generator(seed, device="cpu"):
    if seed is None:
        return None
    normalized_device = _normalize_device(device)
    generator_device = (
        normalized_device if normalized_device.type == "cuda" else torch.device("cpu")
    )
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _sample_without_replacement(
    shape, population_size, sample_size, generator, device
):
    random_scores = torch.rand(
        *shape,
        population_size,
        generator=generator,
        device=device,
    )
    return random_scores.topk(k=sample_size, dim=-1, sorted=False).indices


def _sample_row_indices(
    *,
    batch_size,
    n_estimators,
    n_train,
    subset_size,
    bootstrap,
    generator,
    device,
):
    if bootstrap:
        row_indices = torch.randint(
            low=0,
            high=n_train,
            size=(batch_size, n_estimators, subset_size),
            generator=generator,
            device=device,
        )
    else:
        row_indices = _sample_without_replacement(
            (batch_size, n_estimators),
            n_train,
            subset_size,
            generator,
            device,
        )
    return row_indices


def build_grande_context(
    *,
    batch_size,
    n_estimators,
    num_features_used,
    selected_variables,
    device,
    seed=None,
):
    device = _normalize_device(device)
    generator = _make_generator(seed, device=device)
    take = min(int(num_features_used), selected_variables)
    if take <= 0:
        raise ValueError("num_features_used must be positive")
    features_by_estimator = torch.zeros(
        batch_size, n_estimators, selected_variables, dtype=torch.long
    )
    feature_mask = torch.zeros(
        batch_size, n_estimators, selected_variables, dtype=torch.bool
    )
    sampled_features = _sample_without_replacement(
        (batch_size, n_estimators),
        int(num_features_used),
        take,
        generator,
        device,
    )
    features_by_estimator[..., :take] = sampled_features
    feature_mask[..., :take] = True

    return {
        "features_by_estimator": features_by_estimator.to(device),
        "feature_mask": feature_mask.to(device),
        "num_features_used": int(num_features_used),
    }


def gather_estimator_features(x, features_by_estimator):
    # x: (n_samples, batch, n_features)
    # features_by_estimator: (batch, estimators, selected_variables)
    x_perm = x.permute(1, 0, 2)
    batch_index = torch.arange(x_perm.shape[0], device=x.device)[:, None, None]
    gathered = x_perm[batch_index, :, features_by_estimator]
    return gathered.permute(3, 0, 1, 2)


def _safe_mean_and_std(values, valid_mask, dim):
    valid_count = valid_mask.sum(dim=dim).clamp_min(1)
    masked_values = torch.where(valid_mask, values, torch.zeros_like(values))
    mean = masked_values.sum(dim=dim) / valid_count

    centered = torch.where(
        valid_mask,
        values - mean.unsqueeze(dim),
        torch.zeros_like(values),
    )
    denom = valid_mask.sum(dim=dim).sub(1).clamp_min(1)
    std = torch.sqrt(centered.square().sum(dim=dim) / denom)
    any_valid = valid_mask.any(dim=dim)
    mean = torch.where(any_valid, mean, torch.zeros_like(mean))
    std = torch.where(any_valid, std, torch.zeros_like(std))
    return mean, std


def build_grande_feature_stats(
    *,
    x_train,
    y_train,
    features_by_estimator,
    feature_mask,
    n_out,
    data_subset_fraction=1.0,
    bootstrap=False,
    seed=None,
):
    # x_train: (n_train, batch, n_features)
    batch_size, n_estimators, selected_variables = features_by_estimator.shape
    stats = torch.zeros(
        batch_size,
        n_estimators,
        selected_variables,
        3 + n_out,
        device=x_train.device,
        dtype=x_train.dtype,
    )
    generator = _make_generator(
        None if seed is None else seed + 1, device=x_train.device
    )
    n_train = x_train.shape[0]
    use_subset = data_subset_fraction < 1.0 or bootstrap

    selected_x = gather_estimator_features(x_train, features_by_estimator).permute(
        1, 2, 0, 3
    )
    if y_train.ndim == 1:
        selected_y = y_train.unsqueeze(0).expand(batch_size, -1)
    else:
        selected_y = y_train.transpose(0, 1)
    selected_y = selected_y.unsqueeze(1).expand(-1, n_estimators, -1)

    if use_subset:
        subset_size = max(4, int(math.ceil(n_train * data_subset_fraction)))
        if not bootstrap:
            subset_size = min(subset_size, n_train)
        row_indices = _sample_row_indices(
            batch_size=batch_size,
            n_estimators=n_estimators,
            n_train=n_train,
            subset_size=subset_size,
            bootstrap=bootstrap,
            generator=generator,
            device=x_train.device,
        )
        selected_x = selected_x.gather(
            dim=2,
            index=row_indices.unsqueeze(-1).expand(-1, -1, -1, selected_variables),
        )
        selected_y = selected_y.gather(dim=2, index=row_indices)

    active_feature_mask = feature_mask.unsqueeze(2)
    valid_mask = active_feature_mask & ~torch.isnan(selected_x)
    mean, std = _safe_mean_and_std(selected_x, valid_mask, dim=2)
    active_feature_mask_float = feature_mask.to(dtype=x_train.dtype)
    missing_rate = (
        torch.isnan(selected_x).to(dtype=x_train.dtype).mean(dim=2)
        * active_feature_mask_float
    )

    stats[..., 0] = mean * active_feature_mask_float
    stats[..., 1] = std * active_feature_mask_float
    stats[..., 2] = missing_rate

    if n_out <= 1:
        return stats

    class_targets = selected_y.long().clamp_min(0).clamp_max(n_out - 1)
    class_one_hot = F.one_hot(class_targets, num_classes=n_out).to(dtype=x_train.dtype)
    valid_float = valid_mask.to(dtype=x_train.dtype)
    clean_x = torch.nan_to_num(selected_x, nan=0.0)
    class_denominator = torch.einsum("betk,betc->bekc", valid_float, class_one_hot)
    class_mean = torch.where(
        class_denominator > 0,
        torch.einsum("betk,betc->bekc", clean_x * valid_float, class_one_hot)
        / class_denominator.clamp_min(1),
        torch.zeros_like(class_denominator),
    )
    stats[..., 3:] = class_mean * active_feature_mask_float.unsqueeze(-1)

    return stats


def grande_forward(
    *,
    x,
    split_values,
    split_index_logits,
    estimator_weights,
    leaf_classes,
    features_by_estimator,
    feature_mask,
    path_identifier_list,
    internal_node_index_list,
    training=False,
    dropout=0.0,
    missing_values=True,
    straight_through=False,
    split_temperature=1.0,
):
    dtype = x.dtype
    x_local = gather_estimator_features(x, features_by_estimator)
    nan_mask = torch.isnan(x_local)
    x_local = torch.nan_to_num(x_local, nan=0.0)

    logits = split_index_logits.masked_fill(
        ~feature_mask.unsqueeze(2), -float("inf")
    )
    split_soft = F.softmax(logits / split_temperature, dim=-1)
    split_hard = one_hot_argmax(logits, dim=-1).to(dtype)
    split_index = st(split_hard, split_soft) if straight_through else split_hard

    s1_sum = torch.einsum("beik,beik->bei", split_values, split_index)
    s2_sum = torch.einsum("sbek,beik->sbei", x_local, split_index)

    node_soft = (F.softsign(s1_sum.unsqueeze(0) - s2_sum) + 1) / 2
    node_hard = torch.round(node_soft)
    node_result = st(node_hard, node_soft) if straight_through else node_hard

    selected_nodes = node_result[..., internal_node_index_list]
    left = selected_nodes
    right = 1.0 - selected_nodes

    if missing_values:
        masked_selected = torch.einsum(
            "sbek,beik->sbei", nan_mask.to(dtype), split_index
        )
        masked_ext = masked_selected[..., internal_node_index_list].to(dtype=torch.bool)
        smaller_prob = node_soft.mean(dim=0)
        smaller_prob_ext = smaller_prob[..., internal_node_index_list]
        left = torch.where(masked_ext, smaller_prob_ext, left)
        right = torch.where(masked_ext, 1.0 - smaller_prob_ext, right)

    path_ids = path_identifier_list.to(dtype=dtype)
    path_factors = (1.0 - path_ids) * left + path_ids * right
    path_probs = torch.exp(torch.sum(torch.log(path_factors.clamp_min(1e-7)), dim=-1))

    estimator_weights_leaf = torch.einsum(
        "bel,sbel->sbe", estimator_weights, path_probs
    )
    estimator_weights_softmax = F.softmax(estimator_weights_leaf, dim=-1)
    if training and dropout > 0.0:
        estimator_weights_softmax = F.dropout(
            estimator_weights_softmax, p=dropout, training=True
        )
        estimator_weights_softmax = estimator_weights_softmax / estimator_weights_softmax.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

    weighted_paths = path_probs * estimator_weights_softmax.unsqueeze(-1)
    per_estimator_logits = torch.einsum(
        "belo,sbel->sbeo", leaf_classes, weighted_paths
    )
    return per_estimator_logits.sum(dim=2)
