"""Verifier-based rejection sampling -> SFT data (+ free preference pairs).

    python -m myllm.distill.reject_sample --config configs/distill.yaml

Each prompt has k teacher candidates (``teacher_gen``). Each candidate gets a score in [0, 1]:
  * code prompts (row has ``tests``): fraction of unit-test asserts passed (verifiable);
  * other prompts: pointwise LLM-judge score /10, optionally blended with
    self-consistency agreement (idea #5).

Two selection modes (``distill.mode``):
  * ``hard``     — classic rejection sampling: keep the best candidate if score >= threshold.
  * ``weighted`` — idea #7, Verifier-Weighted Distillation: keep the top-m candidates
    with ``weight = score`` (sampled proportionally by WeightedSamplerMixin), so partially
    correct but informative answers still teach something, with less pull.

Best-vs-worst pairs with a score gap >= ``distill.pair_margin`` are written as DPO data:
the verifier gives preference data for free.
"""

from __future__ import annotations

import argparse
import re

from myllm.clients import make_client
from myllm.config import add_common_args, get, load_config, resolve_path
from myllm.eval.code_exec import pass_fraction
from myllm.eval.judge import judge_score
from myllm.research.self_consistency import agreement_scores, combine
from myllm.utils import read_jsonl, write_jsonl

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def score_candidates(row: dict, judge=None, use_consistency: bool = True) -> list[float]:
    cands = row["candidates"]
    if row.get("tests"):
        return [pass_fraction(c, row["tests"]) for c in cands]
    agree, _unc = agreement_scores(cands) if use_consistency else ([0.0] * len(cands), 0.0)
    if judge is None:
        return agree
    prompt = row["prompt"][-1]["content"]
    judged = []
    for c in cands:
        s = judge_score(judge, prompt, _THINK.sub("", c), reference=row.get("reference"))
        judged.append(None if s is None else min(1.0, max(0.0, s / 10.0)))
    return combine(agree, judged, w_agree=0.3 if use_consistency else 0.0)


def select(row: dict, scores: list[float], mode: str, threshold: float, top_m: int, keep_think: bool) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    picked = order[:1] if mode == "hard" else order[:top_m]
    out = []
    for i in picked:
        if scores[i] < threshold:
            continue
        answer = row["candidates"][i] if keep_think else _THINK.sub("", row["candidates"][i]).strip()
        out.append({
            "id": f"{row.get('id', 'x')}-c{i}",
            "messages": [*row["prompt"], {"role": "assistant", "content": answer}],
            "weight": round(float(scores[i]), 4) if mode == "weighted" else 1.0,
            "source": f"distill:{row.get('teacher', '?')}",
            "thinking_mode": row.get("thinking_mode"),
        })
    return out


def preference_pair(row: dict, scores: list[float], margin: float) -> dict | None:
    if len(scores) < 2:
        return None
    best = max(range(len(scores)), key=lambda i: scores[i])
    worst = min(range(len(scores)), key=lambda i: scores[i])
    if scores[best] - scores[worst] < margin:
        return None
    return {
        "id": f"{row.get('id', 'x')}-pair",
        "prompt": row["prompt"],
        "chosen": [{"role": "assistant", "content": _THINK.sub("", row["candidates"][best]).strip()}],
        "rejected": [{"role": "assistant", "content": _THINK.sub("", row["candidates"][worst]).strip()}],
        "weight": round(scores[best] - scores[worst], 4),
    }


def run(rows: list[dict], cfg: dict, judge=None) -> tuple[list[dict], list[dict]]:
    mode = get(cfg, "distill.mode", "hard")
    sft, pairs = [], []
    for row in rows:
        scores = score_candidates(row, judge, get(cfg, "distill.self_consistency", True))
        sft += select(row, scores, mode, float(get(cfg, "distill.threshold", 0.7)),
                      int(get(cfg, "distill.top_m", 2)), bool(get(cfg, "distill.keep_think", False)))
        pair = preference_pair(row, scores, float(get(cfg, "distill.pair_margin", 0.5)))
        if pair:
            pairs.append(pair)
    return sft, pairs


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="verifier-based rejection sampling"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    rows = read_jsonl(resolve_path(get(cfg, "distill.candidates_out", "data/processed/teacher_candidates.jsonl")))
    judge = make_client(get(cfg, "judge")) if get(cfg, "judge") else None
    sft, pairs = run(rows, cfg, judge)
    n1 = write_jsonl(get(cfg, "distill.sft_out", "data/processed/sft_distilled.jsonl"), sft)
    n2 = write_jsonl(get(cfg, "distill.pairs_out", "data/processed/pref_distilled.jsonl"), pairs)
    print(f"kept {n1} SFT examples and {n2} preference pairs from {len(rows)} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
