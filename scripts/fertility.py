"""Measure tokenizer fertility on Persian (roadmap §5.1) — a 5-minute test that informs the base choice.

    python scripts/fertility.py                                  # default candidate list (needs HF access)
    python scripts/fertility.py --models Qwen/Qwen3-4B-Instruct-2507 google/gemma-3-4b-it --file my_text.txt
    python scripts/fertility.py --models out/tinygpt/tokenizer   # your own tokenizer

Lower tokens/word = more Persian fits in the context window, faster and cheaper inference.
"""

from __future__ import annotations

import argparse
import json

from myllm.tiny_models import resolve_model_name
from myllm.tokenizer_utils import fertility
from myllm.utils import iter_jsonl

DEFAULT_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "CohereLabs/aya-expanse-8b",
]


def load_texts(path: str | None, lang: str | None) -> list[str]:
    if path is None:
        path, lang = "data/samples/pretrain.jsonl", lang or "fa"
    if path.endswith(".jsonl"):
        return [r["text"] for r in iter_jsonl(path) if "text" in r and (lang is None or r.get("domain") == lang)]
    return [t for t in open(path, encoding="utf-8").read().split("\n") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--file", help=".txt (one paragraph per line) or .jsonl with a text field")
    p.add_argument("--lang", help="filter jsonl rows by their 'domain' field (fa/en/code)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from transformers import AutoTokenizer

    texts = load_texts(a.file, a.lang)
    rows = []
    for name in a.models:
        try:
            tok = AutoTokenizer.from_pretrained(resolve_model_name(name))
        except Exception as exc:  # gated / offline / typo
            rows.append({"model": name, "error": str(exc).splitlines()[0][:120]})
            continue
        rows.append({"model": name, "vocab": len(tok), **{k: round(v, 3) for k, v in fertility(tok, texts).items()}})
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'model':<48} {'vocab':>8} {'tok/word':>9} {'chars/tok':>10}")
    for r in sorted(rows, key=lambda r: r.get("tokens_per_word", 1e9)):
        if "error" in r:
            print(f"{r['model']:<48} ERROR {r['error']}")
        else:
            print(f"{r['model']:<48} {r['vocab']:>8} {r['tokens_per_word']:>9.3f} {r['chars_per_token']:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
