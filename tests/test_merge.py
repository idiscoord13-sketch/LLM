import torch

from myllm.train.merge import (
    dare,
    linear_merge,
    merge_state_dicts,
    slerp,
    ties_merge,
    trim_topk,
)


def test_trim_topk_keeps_largest():
    d = torch.tensor([0.1, -5.0, 0.2, 3.0])
    assert torch.equal(trim_topk(d, 0.5), torch.tensor([0.0, -5.0, 0.0, 3.0]))


def test_ties_sign_election_drops_disagreeing_entries():
    d1 = torch.tensor([1.0, 2.0, -1.0])
    d2 = torch.tensor([3.0, -1.0, -3.0])
    merged = ties_merge([d1, d2], [1.0, 1.0], density=1.0)
    # entry0: both +, mean 2; entry1: sign +(2-1) -> only d1 agrees -> 2; entry2: both - -> -2
    assert torch.allclose(merged, torch.tensor([2.0, 2.0, -2.0]))


def test_dare_rescales_in_expectation():
    g = torch.Generator().manual_seed(0)
    d = torch.ones(200_000)
    out = dare(d, 0.3, g)
    assert abs(out.mean().item() - 1.0) < 0.02
    assert abs((out != 0).float().mean().item() - 0.3) < 0.01


def test_slerp_endpoints_and_linear():
    a, b = torch.randn(10), torch.randn(10)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)
    assert torch.allclose(linear_merge([a, b], [1, 1]), (a + b) / 2)


def test_merge_state_dicts_identity_when_models_equal_base():
    base = {"w": torch.randn(4, 4), "step": torch.tensor(3)}
    out = merge_state_dicts(base, [dict(base), dict(base)], "ties", [0.5, 0.5])
    assert torch.allclose(out["w"], base["w"]) and torch.equal(out["step"], base["step"])
