import torch

from ticl.models.decoders import GrandeDecoder
from ticl.models.grande_core import (
    build_grande_context,
    build_tree_index_tensors,
    grande_forward,
)


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
