"""Append-only experiment registry: every run that might support a claim is recorded.

``experiments/registry.jsonl`` rows: {experiment, variant, seed, overrides, metric, value,
git_commit, time, output_dir}. Never delete rows — negative results are results.
"""

from __future__ import annotations

import time
from pathlib import Path

from myllm.config import REPO_ROOT
from myllm.utils import append_jsonl, git_commit, iter_jsonl

REGISTRY = REPO_ROOT / "experiments" / "registry.jsonl"


def record(experiment: str, variant: str, seed: int, metric: str, value: float, overrides: list[str],
           output_dir: str, path: Path = REGISTRY, **extra) -> dict:
    row = {"experiment": experiment, "variant": variant, "seed": seed, "metric": metric, "value": value,
           "overrides": overrides, "output_dir": output_dir, "git_commit": git_commit(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    append_jsonl(path, row)
    return row


def load(experiment: str | None = None, path: Path = REGISTRY) -> list[dict]:
    if not Path(path).exists():
        return []
    return [r for r in iter_jsonl(path) if experiment is None or r["experiment"] == experiment]
