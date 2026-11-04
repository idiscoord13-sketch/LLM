"""Offline stand-ins for real checkpoints, used by smoke configs and tests.

``tiny://qwen3`` resolves to a randomly initialised Qwen3 architecture (same code
path as Qwen/Qwen3-8B, just ~100k parameters) with a ChatML BPE tokenizer trained
on ``data/samples``. This lets every trainer (SFT/DPO/KTO/GKD/GRPO) run end-to-end
on a CPU with no internet, so format/shape bugs are caught before paying for a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from myllm.config import REPO_ROOT

TINY_ROOT = REPO_ROOT / "out" / "tiny"
SAMPLES = REPO_ROOT / "data" / "samples"


def _sample_texts() -> list[str]:
    texts: list[str] = []
    for path in sorted(SAMPLES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts.append(json.dumps(row, ensure_ascii=False))
                if "text" in row:
                    texts.append(row["text"])
    return texts


def tiny_tokenizer(vocab_size: int = 600):
    from transformers import AutoTokenizer

    path = TINY_ROOT / "tokenizer"
    if not (path / "tokenizer.json").exists():
        from myllm.tokenizer_utils import build_tokenizer

        build_tokenizer(_sample_texts(), path, vocab_size=vocab_size, min_frequency=1)
    return AutoTokenizer.from_pretrained(path)


def tiny_model_dir(kind: str = "qwen3", hidden: int = 64, layers: int = 2, seed: int = 0) -> Path:
    """Build (once) and return a directory loadable with ``from_pretrained``."""
    path = TINY_ROOT / f"{kind}-h{hidden}-l{layers}-s{seed}"
    if (path / "config.json").exists():
        return path
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        bos_token_id=None,
    )
    model = Qwen3ForCausalLM(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def resolve_model_name(name: str) -> str:
    """Map ``tiny://qwen3[-teacher]`` to a local directory; pass real HF ids through."""
    if not name.startswith("tiny://"):
        return name
    spec = name[len("tiny://") :]
    if spec == "qwen3":
        return str(tiny_model_dir("qwen3", hidden=64, layers=2, seed=0))
    if spec == "qwen3-teacher":
        return str(tiny_model_dir("qwen3", hidden=96, layers=3, seed=1))
    raise ValueError(f"unknown tiny model {name!r}")
