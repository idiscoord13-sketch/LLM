"""Statistics so that "model B is better" is a claim, not a vibe.

* ``bootstrap_ci``        — percentile CI of a mean (pass@1, accuracy, win-rate...)
* ``paired_bootstrap``    — CI of the *difference* between two systems on the same items
* ``sign_test``           — exact two-sided binomial test on wins vs losses (ties dropped)
* ``seed_summary``        — mean ± std over training seeds (run every ablation with >= 3 seeds)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> dict[str, float]:
    """CI for mean(b - a) over the same items; ``p_le_0`` ~ probability b is not better."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    assert a_arr.shape == b_arr.shape, "paired test needs the same items"
    diff = b_arr - a_arr
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(diff.mean()), "lo": float(lo), "hi": float(hi), "p_le_0": float((means <= 0).mean())}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided p-value that wins and losses are equally likely."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def win_rate_ci(outcomes: Sequence[str], **kw) -> tuple[float, float, float]:
    """Outcomes in {"A","B","tie"} from judge_pair(new, old): A=new wins; ties count 0.5."""
    vals = [1.0 if o == "A" else 0.5 if o == "tie" else 0.0 for o in outcomes]
    return bootstrap_ci(vals, **kw)


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}
