import pytest

from myllm.config import apply_overrides, deep_merge, get, load_config


def test_deep_merge_nested():
    a = {"model": {"name": "x", "dtype": "bf16"}, "seed": 1}
    b = {"model": {"name": "y"}}
    assert deep_merge(a, b) == {"model": {"name": "y", "dtype": "bf16"}, "seed": 1}
    assert a["model"]["name"] == "x"  # not mutated


def test_overrides_parse_yaml_scalars():
    cfg = apply_overrides({}, ["train.learning_rate=1e-4", "train.packing=true", "data.files=[a,b]"])
    assert cfg["train"]["learning_rate"] == 1e-4
    assert cfg["train"]["packing"] is True
    assert cfg["data"]["files"] == ["a", "b"]
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_profile_overrides_recipe_and_cli_wins():
    cfg = load_config("configs/sft.yaml", profile="cpu_smoke", overrides=["train.max_length=99"])
    assert cfg["profile"] == "cpu_smoke"
    assert get(cfg, "model.name") == "tiny://qwen3"
    assert get(cfg, "train.learning_rate") == 2e-4  # from the recipe
    assert get(cfg, "train.max_length") == 99  # CLI beats profile


@pytest.mark.parametrize("profile", ["cpu_smoke", "local_4060", "cloud_24gb", "cloud_80gb"])
def test_all_profiles_load(profile):
    cfg = load_config("configs/sft.yaml", profile=profile)
    assert get(cfg, "model.name")
