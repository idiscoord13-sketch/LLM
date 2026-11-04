"""Idea #5 — Self-Consistency Reward Bootstrapping.

A cheap Persian reward when no good reward model exists: sample k answers; answers
that agree with the majority get higher reward. Agreement is measured on the
*final answer* when one can be extracted (numbers, short spans, code behaviour
outside this module) and on character-n-gram similarity otherwise.

``uncertainty`` (1 - max cluster share) routes the most ambiguous prompts to human
review — active learning that spends the human budget where it matters most.

Known failure mode (documented in docs/research/idea-5.md): on subjective/creative
prompts "agreement" is meaningless and a confident majority can be wrong — combine
with a calibrated judge and keep the ablation honest.
"""

from __future__ import annotations

import re
from collections import Counter

from myllm.data.dedup import jaccard
from myllm.data.normalize_fa import to_ascii_digits

_NUMBER = re.compile(r"-?\d+(?:[.,/]\d+)?")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def final_answer(text: str) -> str | None:
    """Last number in the answer (math-style prompts), else None."""
    nums = _NUMBER.findall(to_ascii_digits(_THINK.sub("", text or "")))
    return nums[-1].replace(",", ".") if nums else None


def agreement_scores(candidates: list[str], sim_threshold: float = 0.5) -> tuple[list[float], float]:
    """Return (per-candidate agreement in [0,1], uncertainty in [0,1])."""
    k = len(candidates)
    if k == 0:
        return [], 1.0
    answers = [final_answer(c) for c in candidates]
    if sum(a is not None for a in answers) >= max(2, k // 2 + 1):
        counts = Counter(a for a in answers if a is not None)
        scores = [counts[a] / k if a is not None else 0.0 for a in answers]
        top_share = counts.most_common(1)[0][1] / k
        return scores, 1.0 - top_share
    # free-form text: mean similarity to the others
    stripped = [_THINK.sub("", c) for c in candidates]
    scores = []
    for i, ci in enumerate(stripped):
        sims = [jaccard(ci, cj, n=4) for j, cj in enumerate(stripped) if j != i]
        scores.append(sum(s >= sim_threshold for s in sims) / max(1, len(sims)))
    return scores, 1.0 - max(scores)


def combine(agreement: list[float], judge: list[float | None], w_agree: float = 0.5) -> list[float]:
    """Blend agreement with a (0..1-normalised) judge score; missing judge -> agreement only."""
    out = []
    for a, j in zip(agreement, judge):
        out.append(a if j is None else w_agree * a + (1 - w_agree) * j)
    return out
