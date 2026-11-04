"""Small shared helpers: seeding, JSONL IO, run bookkeeping."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "?"))
        except Exception:
            continue
    return versions


def save_run_config(out_dir: str | Path, cfg: dict, extra: dict[str, Any] | None = None) -> Path:
    """Persist everything needed to reproduce a run next to its outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "versions": library_versions(),
        **(extra or {}),
    }
    path = out_dir / "run_config.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"config": cfg, "meta": meta}, fh, allow_unicode=True, sort_keys=False)
    return path


def device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"
