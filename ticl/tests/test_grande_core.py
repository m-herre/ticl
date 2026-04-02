import numpy as np
import pytest
import torch

from ticl.models.decoders import GrandeDecoder
from ticl.models.mothernet import MotherNet
from ticl.models.grande_core import (
    build_grande_context,
    build_grande_feature_stats,
    build_tree_index_tensors,
    flatten_grande_estimator_outputs,
    grande_forward,
    pairwise_cosine_off_diag,
)
from ticl.prediction.mothernet import extract_grande_model, predict_with_grande_model


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


def test_grande_combined_output_vectors_capture_estimator_similarity():
    identical_vectors = flatten_grande_estimator_outputs(
        split_values=torch.tensor([[[[1.0]], [[1.0]]]]),
        split_index_logits=torch.tensor([[[[0.0]], [[0.0]]]]),
        estimator_weights=torch.tensor([[[2.0, 3.0], [2.0, 3.0]]]),
        leaf_classes=torch.tensor([[[[4.0]], [[4.0]]]]),
    )
    identical_off_diag = pairwise_cosine_off_diag(identical_vectors)
    assert identical_off_diag is not None
    assert torch.allclose(identical_off_diag, torch.ones_like(identical_off_diag))

    diverse_vectors = flatten_grande_estimator_outputs(
        split_values=torch.tensor([[[[1.0]], [[0.0]]]]),
        split_index_logits=torch.tensor([[[[0.0]], [[1.0]]]]),
        estimator_weights=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
        leaf_classes=torch.tensor([[[[0.0]], [[0.0]]]]),
    )
    diverse_off_diag = pairwise_cosine_off_diag(diverse_vectors)
    assert diverse_off_diag is not None
    assert torch.allclose(
        diverse_off_diag,
        torch.zeros_like(diverse_off_diag),
        atol=1e-6,
    )


def test_flatten_grande_estimator_outputs_casts_all_parts_to_float32():
    combined = flatten_grande_estimator_outputs(
        split_values=torch.randn(2, 3, 4, 1, dtype=torch.float32),
        split_index_logits=torch.randn(2, 3, 4, 1, dtype=torch.bfloat16),
        estimator_weights=torch.randn(2, 3, 2, dtype=torch.float16),
        leaf_classes=torch.randn(2, 3, 4, 3, dtype=torch.bfloat16),
    )

    assert combined.dtype == torch.float32


@torch.no_grad()
@torch.inference_mode()
def test_grande_decoder_outputs_expected_shapes():
    for variant in ("baseline", "factorized_stats", "depthwise_factorized_stats"):
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
            grande_decoder_variant=variant,
            grande_output_init="default",
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


@torch.no_grad()
@torch.inference_mode()
def test_factorized_grande_decoder_zero_delta_reduces_to_feature_means():
    for variant in ("factorized_stats", "depthwise_factorized_stats"):
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
            grande_decoder_variant=variant,
            grande_output_init="default",
        )
        if variant == "factorized_stats":
            for parameter in decoder.split_value_head.parameters():
                parameter.zero_()
        else:
            for parameter in decoder.depthwise_split_value_head.parameters():
                parameter.zero_()

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
        feature_stats = build_grande_feature_stats(
            x_train=x_train,
            y_train=y,
            features_by_estimator=context["features_by_estimator"],
            feature_mask=context["feature_mask"],
            n_out=decoder.n_out,
            data_subset_fraction=decoder.data_subset_fraction,
            bootstrap=decoder.bootstrap,
            seed=0,
        )
        expected_split_values = feature_stats[..., 0].unsqueeze(2).expand_as(
            split_values
        )

        assert split_values.shape == (2, 4, 3, 6)
        assert split_index_logits.shape == (2, 4, 3, 6)
        assert estimator_weights.shape == (2, 4, 4)
        assert leaf_classes.shape == (2, 4, 4, 3)
        assert torch.allclose(split_values, expected_split_values, atol=1e-6)


@torch.no_grad()
@torch.inference_mode()
def test_factorized_grande_decoder_default_init_emits_non_zero_outputs():
    for variant in ("factorized_stats", "depthwise_factorized_stats"):
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
            grande_decoder_variant=variant,
            grande_output_init="default",
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
        outputs = decoder(
            x,
            y,
            x_train,
            context,
            seed=0,
        )
        assert any(output.abs().sum().item() > 0 for output in outputs)


def test_grande_diversity_loss_requires_factorized_variant():
    with pytest.raises(
        ValueError,
        match="grande_diversity_loss_weight requires grande_decoder_variant='factorized_stats'",
    ):
        MotherNet(
            n_out=3,
            emsize=16,
            nhead=4,
            nhid_factor=2,
            nlayers=1,
            n_features=10,
            child_model="grande",
            decoder_type="average",
            decoder_hidden_layers=1,
            decoder_hidden_size=32,
            y_encoder_layer=None,
            tabpfn_zero_weights=False,
            tree_depth=2,
            n_estimators=3,
            selected_variables=4,
            grande_decoder_variant="depthwise_factorized_stats",
            grande_diversity_loss_weight=0.1,
        )


def test_grande_forward_return_aux_reports_diversity_loss():
    model = MotherNet(
        n_out=3,
        emsize=16,
        nhead=4,
        nhid_factor=2,
        nlayers=1,
        n_features=5,
        child_model="grande",
        decoder_type="class_average",
        decoder_hidden_layers=1,
        decoder_hidden_size=32,
        y_encoder_layer=None,
        tabpfn_zero_weights=False,
        tree_depth=2,
        n_estimators=3,
        selected_variables=4,
        grande_decoder_variant="factorized_stats",
        grande_diversity_loss_weight=0.25,
    )
    x = torch.randn(6, 2, 5)
    y = torch.randint(0, 3, (6, 2))

    output_only = model(({"num_features_used": 5}, x, y), single_eval_pos=4)
    output_with_aux, aux_losses = model(
        ({"num_features_used": 5}, x, y),
        single_eval_pos=4,
        return_aux=True,
    )

    assert output_only.shape == output_with_aux.shape
    assert "grande_diversity_loss" in aux_losses
    assert "grande_diversity_cosine_mean" in aux_losses
    assert "grande_diversity_positive_cosine_mean" in aux_losses
    assert aux_losses["grande_diversity_loss"].item() >= 0.0


def test_grande_compile_uses_torch_compile_dynamic(monkeypatch):
    compile_calls = {}

    def fake_compile(fn, dynamic):
        compile_calls["fn"] = fn
        compile_calls["dynamic"] = dynamic
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    model = MotherNet(
        n_out=3,
        emsize=16,
        nhead=4,
        nhid_factor=2,
        nlayers=1,
        n_features=5,
        child_model="grande",
        decoder_type="class_average",
        decoder_hidden_layers=1,
        decoder_hidden_size=32,
        y_encoder_layer=None,
        tabpfn_zero_weights=False,
        tree_depth=2,
        n_estimators=3,
        selected_variables=4,
        grande_decoder_variant="factorized_stats",
        grande_compile=True,
    )

    assert compile_calls["fn"] is grande_forward
    assert compile_calls["dynamic"] is True
    assert model._grande_forward is grande_forward


def test_grande_compile_falls_back_when_torch_compile_is_unavailable(monkeypatch):
    monkeypatch.delattr(torch, "compile", raising=False)

    with pytest.warns(RuntimeWarning, match="torch.compile is unavailable"):
        model = MotherNet(
            n_out=3,
            emsize=16,
            nhead=4,
            nhid_factor=2,
            nlayers=1,
            n_features=5,
            child_model="grande",
            decoder_type="class_average",
            decoder_hidden_layers=1,
            decoder_hidden_size=32,
            y_encoder_layer=None,
            tabpfn_zero_weights=False,
            tree_depth=2,
            n_estimators=3,
            selected_variables=4,
            grande_decoder_variant="factorized_stats",
            grande_compile=True,
        )

    assert model._grande_forward is grande_forward


@torch.no_grad()
@torch.inference_mode()
def test_depthwise_grande_decoder_tracks_temperature_steps_during_training():
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
        grande_decoder_variant="depthwise_factorized_stats",
        grande_output_init="default",
        grande_split_temperature_start=2.0,
        grande_split_temperature_end=1.0,
        grande_split_temperature_anneal_steps=4,
    )
    decoder.train()
    x = torch.randn(5, 2, 32)
    y = torch.randint(0, 3, (5, 2))
    x_train = torch.randn(5, 2, 10)
    context = decoder.build_context(
        batch_size=2,
        num_features_used=4,
        device=x.device,
        seed=0,
    )
    decoder(
        x,
        y,
        x_train,
        context,
        seed=0,
    )
    decoder(
        x,
        y,
        x_train,
        context,
        seed=0,
    )
    assert decoder.split_temperature_step.item() == 2


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


@torch.no_grad()
@torch.inference_mode()
def test_grande_extract_and_predict_smoke():
    torch.manual_seed(0)
    np.random.seed(0)

    for variant in ("factorized_stats", "depthwise_factorized_stats"):
        model = MotherNet(
            n_out=3,
            emsize=16,
            nhead=4,
            nhid_factor=2,
            nlayers=1,
            n_features=10,
            child_model="grande",
            decoder_type="average",
            decoder_hidden_layers=1,
            decoder_hidden_size=32,
            y_encoder_layer=None,
            tabpfn_zero_weights=False,
            tree_depth=2,
            n_estimators=3,
            selected_variables=4,
            grande_decoder_variant=variant,
            grande_output_init="default",
            grande_split_temperature_start=1.0,
            grande_split_temperature_end=0.5,
            grande_split_temperature_anneal_steps=4,
        )
        model.eval()
        config = {
            "prior": {"num_features": 10},
            "mothernet": {"grande_random_state": 7},
        }
        x_train = np.array(
            [
                [0.1, -1.0, 0.3, 1.2],
                [0.4, -0.5, -0.2, 0.9],
                [1.0, 0.2, 0.5, -0.4],
                [-0.3, 0.7, -0.8, 0.1],
                [0.6, -0.1, 1.1, -0.7],
                [0.2, 0.4, -0.6, 0.5],
            ],
            dtype=float,
        )
        y_train = np.array([0, 1, 2, 1, 0, 2], dtype=int)
        x_test = np.array(
            [
                [0.3, -0.2, 0.4, 0.8],
                [-0.1, 0.5, -0.7, 0.0],
                [0.9, 0.1, 0.2, -0.5],
            ],
            dtype=float,
        )

        grande_params = extract_grande_model(
            model,
            config,
            x_train,
            y_train,
            device="cpu",
            inference_device="cpu",
            scale=True,
        )
        assert grande_params["path_identifier_list"].dtype == np.float32
        train_mean = np.nan_to_num(np.nanmean(x_train, axis=0), 0.0)
        train_std = np.nanstd(x_train, axis=0, ddof=1) + 0.000001
        train_std[np.isnan(train_std)] = 1.0
        probs = predict_with_grande_model(
            train_mean,
            train_std,
            x_test,
            grande_params,
            scale=True,
            inference_device="cpu",
            n_classes=3,
        )

        assert probs.shape == (3, 3)
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
