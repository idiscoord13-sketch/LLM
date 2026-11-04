"""Download capped samples of the open datasets in configs/data_sources.yaml (needs internet + HF).

    python scripts/download_data.py --only naab fineweb_edu code          # tiny-GPT corpora
    python scripts/download_data.py --only farsinstruct tulu3_sft --max-rows 20000
    huggingface-cli login     # first, for gated datasets (CulturaX, the-stack-smol)

Rows are written as JSONL in the repo's formats: {"text"} for raw text, chat
{"messages"} for SFT, {"prompt","chosen","rejected"} for preferences,
{"prompt","tests"} for code tasks. Run the cleaning pipeline afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myllm.config import load_yaml
from myllm.data.formats import to_messages


def convert(row: dict, fmt: str, text_field: str | None) -> dict | None:
    if fmt in (None, "text"):
        text = row.get(text_field or "text")
        return {"text": text} if isinstance(text, str) and text.strip() else None
    if fmt == "preference":
        return {"prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [m for m in row["chosen"] if m["role"] == "assistant"][-1:],
                "rejected": [m for m in row["rejected"] if m["role"] == "assistant"][-1:]}
    if fmt == "mbpp":
        return {"id": f"mbpp-{row['task_id']}", "prompt": [{"role": "user", "content": row["text"]}],
                "tests": "\n".join(row["test_list"])}
    try:
        return to_messages(row)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data_sources.yaml")
    p.add_argument("--only", nargs="*")
    p.add_argument("--max-rows", type=int)
    a = p.parse_args(argv)
    from datasets import load_dataset

    sources = load_yaml(a.config)["sources"]
    for name, src in sources.items():
        if a.only and name not in a.only:
            continue
        if "out" not in src:
            print(f"skip {name} (eval-only; use lm-evaluation-harness / its own loader)")
            continue
        kwargs = {k: src[k] for k in ("data_dir",) if k in src}
        ds = load_dataset(src["hf"], src.get("config"), split=src.get("split", "train"), streaming=True, **kwargs)
        limit = a.max_rows or src.get("max_rows", 10000)
        out = Path(src["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w", encoding="utf-8") as fh:
            for row in ds:
                conv = convert(row, src.get("format", "text"), src.get("text_field"))
                if conv is None:
                    continue
                conv.update({"source": name, "license": src.get("license")})
                fh.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n += 1
                if n >= limit:
                    break
        print(f"{name}: {n} rows -> {out} (license: {src.get('license')}{' — VERIFY' if src.get('verify') else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
