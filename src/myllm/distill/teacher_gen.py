"""Offline distillation: a large open teacher writes k candidate answers per prompt.

    python -m myllm.distill.teacher_gen --config configs/distill.yaml

Output rows (one per prompt)::

    {"id", "prompt": [...messages], "candidates": ["...", ...], "tests"?: "...", "teacher": "..."}

Then ``myllm.distill.reject_sample`` scores the candidates with verifiers
(unit tests for code, LLM judge for prose) and writes SFT data.

Licensing: use an open-weight teacher whose license allows training other models
(Qwen = Apache-2.0). Many closed APIs forbid using outputs to train competitors.

Idea #6 (Cross-Lingual Thinking Bridge) is a *prompting mode* here:
``thinking_mode: en_think_fa_answer`` asks the teacher to reason in English inside
<think>...</think> and answer in Persian; compare against ``fa_think`` and ``no_think``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from myllm.clients import make_client
from myllm.config import add_common_args, get, load_config, resolve_path
from myllm.utils import read_jsonl, write_jsonl

THINKING_SYSTEM = {
    "no_think": "تو یک دستیار دقیق و مفید هستی. به فارسی روان و دقیق پاسخ بده. اگر سؤال برنامه‌نویسی است، کد درست و تست‌شده در یک بلوک ```python بنویس.",
    "fa_think": "تو یک دستیار دقیق هستی. ابتدا گام‌به‌گام و به فارسی داخل <think>...</think> فکر کن، سپس پاسخ نهایی را به فارسی روان بنویس. کد را در یک بلوک ```python بنویس.",
    "en_think_fa_answer": "You are a precise assistant. First reason step by step IN ENGLISH inside <think>...</think>. Then write the final answer IN FLUENT PERSIAN (Farsi). Put any code in a single ```python block.",
}


def build_messages(prompt: list[dict], mode: str) -> list[dict]:
    if prompt and prompt[0]["role"] == "system":
        return prompt
    return [{"role": "system", "content": THINKING_SYSTEM[mode]}, *prompt]


def generate(rows: list[dict], client, k: int, mode: str, params: dict, workers: int = 8) -> list[dict]:
    def one(row: dict) -> dict:
        prompt = row["prompt"] if isinstance(row["prompt"], list) else [{"role": "user", "content": row["prompt"]}]
        cands = client.chat(build_messages(prompt, mode), n=k, **params)
        return {**row, "prompt": prompt, "candidates": cands, "thinking_mode": mode,
                "teacher": getattr(client, "model", "?")}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="teacher candidate generation"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    rows = []
    for f in get(cfg, "data.prompt_files", []):
        rows.extend(read_jsonl(resolve_path(f)))
    if args.smoke:
        rows = rows[:10]
    client = make_client(get(cfg, "teacher"))
    out = generate(rows, client, k=int(get(cfg, "distill.k", 4)), mode=get(cfg, "distill.thinking_mode", "no_think"),
                   params=get(cfg, "distill.params", {"temperature": 0.8, "max_tokens": 2048}),
                   workers=int(get(cfg, "distill.workers", 8)))
    n = write_jsonl(get(cfg, "distill.candidates_out", "data/processed/teacher_candidates.jsonl"), out)
    print(f"wrote {n} prompts x {get(cfg, 'distill.k', 4)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
