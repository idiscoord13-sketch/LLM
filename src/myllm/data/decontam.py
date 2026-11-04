"""Benchmark decontamination (roadmap §4.2 step 6, §8.3).

Any training row sharing a word n-gram with an eval item is dropped. Default
n=8 (eval prompts are short; 13 is the classic value for long benchmark docs).
Run this against the golden set *and* every benchmark you report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from myllm.data.normalize_fa import normalize_fa

_TOKEN = re.compile(r"[\w‌]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.replace("‌", "") for t in _TOKEN.findall(normalize_fa(text, digits="ascii").lower())]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class Decontaminator:
    def __init__(self, eval_texts: Iterable[str], n: int = 8, min_eval_tokens: int = 6):
        self.n = n
        self.index: set[tuple[str, ...]] = set()
        for text in eval_texts:
            if len(_tokens(text)) >= min_eval_tokens:  # very short items would match everything
                self.index |= ngrams(text, n)

    def is_contaminated(self, text: str) -> bool:
        if not self.index:
            return False
        return not ngrams(text, self.n).isdisjoint(self.index)
