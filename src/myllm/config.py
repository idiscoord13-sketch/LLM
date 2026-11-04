"""YAML config loading with hardware profiles and CLI overrides.

Resolution order (later wins):
    1. the recipe config file (e.g. configs/sft.yaml)
    2. the hardware profile (configs/profiles/<name>.yaml) — chosen by
       ``--profile`` or by the ``profile:`` key inside the recipe
    3. ``--set dotted.key=value`` overrides from the command line

A recipe says *what* to train (lr, epochs, LoRA rank...). A profile says *where*
(model size, batch, sequence length, 4-bit or not). Keeping them apart lets the
same recipe run on an 8GB RTX 4060 today and an 80GB cloud GPU tomorrow.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse ``--set`` values with YAML rules so ``1e-4``, ``true``, ``[1,2]`` work."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str):  # YAML 1.1 reads "1e-4" as a string; accept it as a float
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {dotted}: {part} is not a mapping")
        node[parts[-1]] = _parse_scalar(raw)
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    candidate = REPO_ROOT / p
    return candidate if candidate.exists() else p


def load_yaml(path: str | Path) -> dict:
    with open(resolve_path(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"unknown profile {name!r}; available: {available}")
    return load_yaml(path)


def load_config(
    path: str | Path,
    profile: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    cfg = load_yaml(path)
    profile_name = profile or cfg.get("profile")
    if profile_name:
        cfg = deep_merge(cfg, load_profile(profile_name))
        cfg["profile"] = profile_name
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="recipe YAML (e.g. configs/sft.yaml)")
    parser.add_argument("--profile", default=None, help="hardware profile in configs/profiles/")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set train.learning_rate=1e-4",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (100 samples, max_steps=10) to catch shape/format bugs before a long run",
    )
    return parser


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
