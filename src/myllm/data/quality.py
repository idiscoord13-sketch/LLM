"""Quality filtering (roadmap §4.2 step 4).

Two layers:
1. ``heuristic_quality`` — Gopher/FineWeb-style rules (length, symbol ratio,
   repetition, mean word length). Cheap, explainable, catches most junk.
2. ``HashedQualityClassifier`` — a FineWeb-Edu-style learned filter: label a few
   thousand documents 0..5 with an LLM judge (``llm_quality_prompt``), train this
   tiny hashed char-n-gram logistic regression (pure numpy, no sklearn), then score
   millions of documents for free.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_WORD = re.compile(r"[\w‌]+", re.UNICODE)
_SYMBOL = re.compile(r"[#{}<>|\\^~`$*_=+@]")


@dataclass
class QualityReport:
    keep: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def _dup_line_fraction(lines: list[str]) -> float:
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup_chars = sum(len(line) * (c - 1) for line, c in counts.items() if c > 1)
    return dup_chars / max(1, sum(len(line) for line in lines))


def _top_ngram_fraction(words: list[str], n: int = 3) -> float:
    if len(words) < n + 2:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return count * sum(len(w) for w in gram) / max(1, sum(len(w) for w in words))


def heuristic_quality(
    text: str,
    *,
    min_chars: int = 30,
    max_chars: int = 200_000,
    min_words: int = 5,
    max_symbol_ratio: float = 0.1,
    max_dup_line_frac: float = 0.3,
    max_top_trigram_frac: float = 0.2,
    mean_word_len: tuple[float, float] = (2.0, 12.0),
    min_unique_word_ratio: float = 0.3,
    min_words_for_repetition: int = 50,
    is_code: bool = False,
) -> QualityReport:
    reasons: list[str] = []
    words = _WORD.findall(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stats = {
        "chars": float(len(text)),
        "words": float(len(words)),
        "symbol_ratio": len(_SYMBOL.findall(text)) / max(1, len(text)),
        "dup_line_frac": _dup_line_fraction(lines),
        "top_trigram_frac": _top_ngram_fraction(words),
        "mean_word_len": sum(len(w) for w in words) / max(1, len(words)),
        "unique_word_ratio": len(set(words)) / max(1, len(words)),
    }
    if stats["chars"] < min_chars:
        reasons.append("too_short")
    if stats["chars"] > max_chars:
        reasons.append("too_long")
    if stats["words"] < min_words:
        reasons.append("too_few_words")
    if not is_code:  # code legitimately has many symbols and repeated lines
        if stats["symbol_ratio"] > max_symbol_ratio:
            reasons.append("symbol_ratio")
        if stats["dup_line_frac"] > max_dup_line_frac:
            reasons.append("duplicate_lines")
        lo, hi = mean_word_len
        if words and not lo <= stats["mean_word_len"] <= hi:
            reasons.append("mean_word_len")
        if len(words) >= 10 and stats["unique_word_ratio"] < min_unique_word_ratio:
            reasons.append("low_vocabulary")  # "buy buy buy cheap cheap" spam
    # n-gram repetition statistics are only meaningful on longer documents (Gopher rules)
    if len(words) >= min_words_for_repetition and stats["top_trigram_frac"] > max_top_trigram_frac:
        reasons.append("repetitive_ngrams")
    return QualityReport(keep=not reasons, reasons=reasons, stats=stats)


def compression_ratio(text: str) -> float:
    """Very repetitive text compresses extremely well; a cheap extra signal."""
    raw = text.encode("utf-8")
    return len(zlib.compress(raw)) / max(1, len(raw))


LLM_QUALITY_PROMPT_FA = """متن زیر را از نظر ارزش آموزشی و کیفیت برای آموزش یک مدل زبانی ارزیابی کن.
معیارها: درستی، انسجام، اطلاعات مفید، نگارش سالم فارسی، نبودِ اسپم/تبلیغ/متن ماشینیِ بی‌معنا.
فقط یک عدد صحیح از ۰ (بی‌ارزش) تا ۵ (عالی و آموزشی) برگردان.

متن:
\"\"\"{text}\"\"\"

نمره:"""


def llm_quality_prompt(text: str, max_chars: int = 3000) -> str:
    return LLM_QUALITY_PROMPT_FA.format(text=text[:max_chars])


class HashedQualityClassifier:
    """Logistic regression on hashed character n-grams (binary: score >= threshold)."""

    def __init__(self, n_features: int = 2**18, ngram: tuple[int, int] = (2, 4), seed: int = 0):
        self.n_features = n_features
        self.ngram = ngram
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.rng = np.random.default_rng(seed)

    def _features(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        t = text.lower()
        idx: Counter = Counter()
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                idx[zlib.crc32(t[i : i + n].encode("utf-8")) % self.n_features] += 1
        if not idx:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        keys = np.fromiter(idx.keys(), dtype=np.int64)
        vals = np.fromiter(idx.values(), dtype=np.float32)
        vals = np.log1p(vals)
        vals /= math.sqrt(float((vals**2).sum())) or 1.0
        return keys, vals

    def predict_proba(self, text: str) -> float:
        keys, vals = self._features(text)
        z = float(self.w[keys] @ vals) + self.b
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def fit(self, texts: list[str], labels: list[int], epochs: int = 5, lr: float = 0.5, l2: float = 1e-6):
        feats = [self._features(t) for t in texts]
        y = np.asarray(labels, dtype=np.float32)
        order = np.arange(len(texts))
        for _ in range(epochs):
            self.rng.shuffle(order)
            for i in order:
                keys, vals = feats[i]
                z = float(self.w[keys] @ vals) + self.b
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g = p - y[i]
                self.w[keys] -= lr * (g * vals + l2 * self.w[keys])
                self.b -= lr * g
        return self

    def save(self, path: str) -> None:
        np.savez_compressed(path, w=self.w, b=np.array([self.b]), meta=np.array([self.n_features, *self.ngram]))

    @classmethod
    def load(cls, path: str) -> HashedQualityClassifier:
        data = np.load(path)
        n_features, lo, hi = (int(x) for x in data["meta"])
        clf = cls(n_features=n_features, ngram=(lo, hi))
        clf.w = data["w"]
        clf.b = float(data["b"][0])
        return clf
