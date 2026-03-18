from ticl.utils import get_wandb_run_string


def test_get_wandb_run_string_keeps_short_names():
    model_string = "mn_childmodelgrande_short"
    assert get_wandb_run_string(model_string) == model_string


def test_get_wandb_run_string_truncates_long_names_deterministically():
    model_string = "mn_" + "verylongsegment_" * 20
    first = get_wandb_run_string(model_string)
    second = get_wandb_run_string(model_string)

    assert first == second
    assert len(first) <= 128
    assert first != model_string
    assert "_" in first
