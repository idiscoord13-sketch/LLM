"""Reward functions for GRPO / RLVR (roadmap §6.4).

TRL calls each function as ``fn(completions=..., prompts=..., **dataset_columns)`` and
expects one float per completion. Completions are either strings or, for
conversational data, ``[{"role": "assistant", "content": ...}]``.

Verifiable rewards beat learned reward models for code/math because they cannot be
"flattered" (no reward hacking via length or tone). We still add small shaping
terms (format, Persian prose, length) — each is separately weighted in the config.
"""

from __future__ import annotations

import re

from myllm.eval.code_exec import pass_fraction

_FENCE = re.compile(r"```(?:python|py)\s*\n.*?```", re.DOTALL | re.IGNORECASE)
_PERSIAN = re.compile("[؀-ۿ]")
_LATIN = re.compile("[A-Za-z]")


def _text(completion) -> str:
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def code_tests_reward(completions, tests=None, **_) -> list[float]:
    """Fraction of unit-test asserts passed (0..1). Needs a ``tests`` dataset column."""
    if tests is None:
        return [0.0] * len(completions)
    return [pass_fraction(_text(c), t or "") for c, t in zip(completions, tests)]


def code_format_reward(completions, **_) -> list[float]:
    """1 if the answer contains exactly one fenced python block."""
    return [1.0 if len(_FENCE.findall(_text(c))) == 1 else 0.0 for c in completions]


def persian_prose_reward(completions, **_) -> list[float]:
    """Share of Persian letters in the prose outside code blocks (keeps answers in Persian)."""
    scores = []
    for c in completions:
        prose = _FENCE.sub("", _text(c))
        fa, la = len(_PERSIAN.findall(prose)), len(_LATIN.findall(prose))
        scores.append(fa / (fa + la) if fa + la else 0.0)
    return scores


def length_penalty(completions, max_chars: int = 4000, **_) -> list[float]:
    """0 up to max_chars, then linearly down to -1 at 2*max_chars (anti verbosity-hacking)."""
    out = []
    for c in completions:
        n = len(_text(c))
        out.append(0.0 if n <= max_chars else -min(1.0, (n - max_chars) / max_chars))
    return out


REWARDS = {
    "code_tests": code_tests_reward,
    "code_format": code_format_reward,
    "persian_prose": persian_prose_reward,
    "length_penalty": length_penalty,
}
