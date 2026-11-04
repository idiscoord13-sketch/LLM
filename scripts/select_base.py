"""Pick the base model by measurement, not by blog posts (roadmap §2, "best of the best").

    python scripts/select_base.py --candidates Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 google/gemma-3-4b-it

For each candidate (served by an OpenAI-compatible endpoint you start, e.g. vLLM/Ollama):
  1. tokenizer fertility on Persian (cheaper context = more knowledge per token)
  2. golden-set metrics (code pass@1, must-include accuracy, safety/over-refusal)
  3. optional: lm-evaluation-harness Persian benchmarks (pass --lm-eval-tasks)
and writes a ranked table to out/select_base/report.md.

Endpoints: --endpoint MODEL=URL pairs, e.g.  --endpoint Qwen/Qwen3-4B-Instruct-2507=http://localhost:8000/v1
Missing endpoints -> only fertility is measured for that model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from myllm.clients import ChatClient
from myllm.eval.run_eval import generate_answers, score
from myllm.tokenizer_utils import fertility
from myllm.utils import read_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--endpoint", action="append", default=[], help="MODEL=BASE_URL")
    p.add_argument("--golden", default="data/eval/golden_fa.jsonl")
    p.add_argument("--fa-text", default="data/samples/pretrain.jsonl")
    p.add_argument("--lm-eval-tasks", default="", help="comma-separated lm-eval task names (Persian benchmarks)")
    p.add_argument("--out", default="out/select_base")
    a = p.parse_args(argv)
    from transformers import AutoTokenizer

    endpoints = dict(e.split("=", 1) for e in a.endpoint)
    fa_texts = [r["text"] for r in read_jsonl(a.fa_text) if r.get("domain", "fa") == "fa"]
    golden = read_jsonl(a.golden)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for name in a.candidates:
        row = {"model": name}
        try:
            row["tok_per_word_fa"] = round(fertility(AutoTokenizer.from_pretrained(name), fa_texts)["tokens_per_word"], 3)
        except Exception as exc:
            row["tok_per_word_fa"] = f"error: {str(exc)[:60]}"
        if name in endpoints:
            client = ChatClient(base_url=endpoints[name], model=name)
            answers = generate_answers(client, golden, None, {"temperature": 0.0, "max_tokens": 1024})
            metrics, _ = score(golden, answers)
            row.update({k: v for k, v in metrics.items() if not k.endswith(("_ci", "_n"))})
        if a.lm_eval_tasks and name in endpoints:
            res = out / f"lm_eval_{name.replace('/', '_')}"
            subprocess.run(["lm_eval", "--model", "local-chat-completions", "--model_args",
                            f"model={name},base_url={endpoints[name]}/chat/completions", "--tasks", a.lm_eval_tasks,
                            "--apply_chat_template", "--output_path", str(res)], check=False)
            row["lm_eval_output"] = str(res)
        table.append(row)
    keys = sorted({k for r in table for k in r} - {"model"})
    md = ["| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    md += ["| " + r["model"] + " | " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in table]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
