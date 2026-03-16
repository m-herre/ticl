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


def _make_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def build_grande_context(
    *,
    batch_size,
    n_estimators,
    num_features_used,
    selected_variables,
    device,
    seed=None,
):
    generator = _make_generator(seed)
    features_by_estimator = torch.zeros(
        batch_size, n_estimators, selected_variables, dtype=torch.long
    )
    feature_mask = torch.zeros(
        batch_size, n_estimators, selected_variables, dtype=torch.bool
    )

    take = min(int(num_features_used), selected_variables)
    if take <= 0:
        raise ValueError("num_features_used must be positive")

    for batch_idx in range(batch_size):
        for estimator_idx in range(n_estimators):
            perm = torch.randperm(int(num_features_used), generator=generator)
            features_by_estimator[batch_idx, estimator_idx, :take] = perm[:take]
            feature_mask[batch_idx, estimator_idx, :take] = True

    return {
        "features_by_estimator": features_by_estimator.to(device),
        "feature_mask": feature_mask.to(device),
        "num_features_used": int(num_features_used),
    }


def gather_estimator_features(x, features_by_estimator):
    # x: (n_samples, batch, n_features)
    # features_by_estimator: (batch, estimators, selected_variables)
    x_perm = x.permute(1, 0, 2)
    gather_index = features_by_estimator.unsqueeze(1).expand(
        x_perm.shape[0], x_perm.shape[1], features_by_estimator.shape[1], features_by_estimator.shape[2]
    )
    x_expanded = x_perm.unsqueeze(2).expand(
        x_perm.shape[0], x_perm.shape[1], features_by_estimator.shape[1], x_perm.shape[2]
    )
    gathered = torch.gather(x_expanded, dim=3, index=gather_index)
    return gathered.permute(1, 0, 2, 3)


def _safe_mean_and_std(values, valid_mask):
    valid_count = valid_mask.sum(dim=0).clamp_min(1)
    masked_values = torch.where(valid_mask, values, torch.zeros_like(values))
    mean = masked_values.sum(dim=0) / valid_count

    centered = torch.where(valid_mask, values - mean.unsqueeze(0), torch.zeros_like(values))
    denom = valid_mask.sum(dim=0).sub(1).clamp_min(1)
    std = torch.sqrt((centered.square().sum(dim=0)) / denom)
    mean = torch.where(valid_mask.any(dim=0), mean, torch.zeros_like(mean))
    std = torch.where(valid_mask.any(dim=0), std, torch.zeros_like(std))
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
    generator = _make_generator(None if seed is None else seed + 1)
    n_train = x_train.shape[0]
    use_subset = data_subset_fraction < 1.0 or bootstrap

    for batch_idx in range(batch_size):
        for estimator_idx in range(n_estimators):
            feature_ids = features_by_estimator[batch_idx, estimator_idx]
            mask = feature_mask[batch_idx, estimator_idx]
            used = int(mask.sum().item())
            if used == 0:
                continue

            selected_x = x_train[:, batch_idx, feature_ids[:used]]
            if y_train.ndim == 1:
                selected_y = y_train
            else:
                selected_y = y_train[:, batch_idx]

            if use_subset:
                subset_size = max(4, int(math.ceil(n_train * data_subset_fraction)))
                if bootstrap:
                    row_indices = torch.randint(
                        low=0,
                        high=n_train,
                        size=(subset_size,),
                        generator=generator,
                    ).to(x_train.device)
                else:
                    subset_size = min(subset_size, n_train)
                    row_indices = torch.randperm(n_train, generator=generator)[
                        :subset_size
                    ].to(x_train.device)
                selected_x = selected_x[row_indices]
                selected_y = selected_y[row_indices]

            valid_mask = ~torch.isnan(selected_x)
            mean, std = _safe_mean_and_std(selected_x, valid_mask)
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
                    torch.zeros(used, device=x_train.device, dtype=x_train.dtype),
                )
                stats[batch_idx, estimator_idx, :used, 3 + class_idx] = class_mean

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
):
    dtype = x.dtype
    x_local = gather_estimator_features(x, features_by_estimator)
    nan_mask = torch.isnan(x_local)
    x_local = torch.nan_to_num(x_local, nan=0.0)

    logits = split_index_logits.masked_fill(
        ~feature_mask.unsqueeze(2), -float("inf")
    )
    split_soft = F.softmax(logits, dim=-1)
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
    path_probs = torch.prod(
        ((1.0 - path_ids) * left + path_ids * right),
        dim=-1,
    )

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
