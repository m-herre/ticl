import tempfile

import pytest
import torch

from ticl.model_builder import get_model, load_model
from ticl.model_configs import get_model_default_config
from ticl.models.grande_core import grande_forward


def _make_small_grande_config():
    config = get_model_default_config("mothernet")
    config["prior"]["num_features"] = 8
    config["prior"]["classification"]["max_num_classes"] = 3
    config["transformer"].update({
        "emsize": 32,
        "nlayers": 2,
        "nhid_factor": 2,
        "nhead": 4,
    })
    config["mothernet"].update({
        "child_model": "grande",
        "decoder_type": "class_average",
        "decoder_embed_dim": 64,
        "decoder_hidden_size": 64,
        "decoder_hidden_layers": 1,
        "tree_depth": 2,
        "n_estimators": 2,
        "selected_variables": 4,
        "grande_decoder_variant": "factorized_stats",
        "grande_split_temperature_start": 1.0,
        "grande_split_temperature_end": 0.5,
        "grande_split_temperature_anneal_steps": 4,
    })
    return config


def _save_checkpoint(path, model_state, config):
    torch.save((model_state, None, None, config), path)


def test_load_model_backfills_missing_grande_buffers():
    config = _make_small_grande_config()
    load_model.cache_clear()
    _, model, *_ = get_model(config, device="cpu", should_train=False)
    model_state = model.state_dict()
    model_state.pop("decoder.split_temperature_step")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/legacy_grande.cpkt"
        _save_checkpoint(checkpoint_path, model_state, config)
        loaded_model, loaded_config = load_model(checkpoint_path, device="cpu")

    assert loaded_config["mothernet"]["child_model"] == "grande"
    assert loaded_model.decoder.split_temperature_step.item() == 0


def test_load_model_still_requires_grande_parameters():
    config = _make_small_grande_config()
    load_model.cache_clear()
    _, model, *_ = get_model(config, device="cpu", should_train=False)
    model_state = model.state_dict()
    model_state.pop("decoder.estimator_embedding.weight")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/broken_grande.cpkt"
        _save_checkpoint(checkpoint_path, model_state, config)
        load_model.cache_clear()
        with pytest.raises(RuntimeError, match="decoder.estimator_embedding.weight"):
            load_model(checkpoint_path, device="cpu")


def test_load_model_grande_checkpoint_stays_portable_without_torch_compile(monkeypatch):
    config = _make_small_grande_config()
    config["mothernet"]["grande_compile"] = True
    load_model.cache_clear()
    _, model, *_ = get_model(config, device="cpu", should_train=False)
    model_state = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/compiled_grande.cpkt"
        _save_checkpoint(checkpoint_path, model_state, config)
        load_model.cache_clear()
        monkeypatch.delattr(torch, "compile", raising=False)
        with pytest.warns(RuntimeWarning, match="torch.compile is unavailable"):
            loaded_model, loaded_config = load_model(checkpoint_path, device="cpu")

    assert loaded_config["mothernet"]["grande_compile"] is True
    assert loaded_model._grande_forward is grande_forward
