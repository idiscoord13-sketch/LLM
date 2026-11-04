"""Language identification (roadmap §4.2 step 2).

Default: a fast, dependency-free script heuristic that separates Persian, Arabic,
English and code. For production-grade langID pass a fastText/GlotLID model path
(``pip install fasttext`` + download ``lid.176.bin`` or GlotLID ``model.bin``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARABIC_SCRIPT = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile("[A-Za-z]")
_PERSIAN_ONLY = set("پچژگکی")  # letters Persian uses but Standard Arabic does not
_ARABIC_ONLY = set("ةيكىإأؤ")  # letters (pre-normalization) typical for Arabic
_PERSIAN_WORDS = {"است", "این", "که", "را", "می", "برای", "با", "از", "به", "هم", "یک", "شود", "کنید"}
_ARABIC_WORDS = {"في", "من", "على", "إلى", "هذا", "الذي", "التي", "كان", "عن", "مع", "هو", "هي"}
_CODE_HINTS = re.compile(
    r"(^\s*(def|class|import|from|return|for|while|if|elif|else|try|except|function|const|let|var|#include|public|private)\b)"
    r"|[{};]\s*$|==|!=|->|=>|\(\)|\bprint\(",
    re.MULTILINE,
)


@dataclass
class LangGuess:
    lang: str  # fa | ar | en | code | mixed | unknown
    score: float  # confidence-ish in [0, 1]


def guess_language(text: str) -> LangGuess:
    if not text or not text.strip():
        return LangGuess("unknown", 0.0)
    arabic_script = len(_ARABIC_SCRIPT.findall(text))
    latin = len(_LATIN.findall(text))
    letters = arabic_script + latin
    if letters == 0:
        return LangGuess("unknown", 0.0)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    code_lines = sum(1 for ln in lines if _CODE_HINTS.search(ln))
    if lines and code_lines / len(lines) >= 0.4 and arabic_script / letters < 0.3:
        return LangGuess("code", code_lines / len(lines))

    ar_ratio = arabic_script / letters
    if ar_ratio >= 0.6:
        words = set(re.findall(r"[؀-ۿ]+", text))
        fa_score = len(set(text) & _PERSIAN_ONLY) + 2 * len(words & _PERSIAN_WORDS)
        ar_score = len(set(text) & _ARABIC_ONLY) + 2 * len(words & _ARABIC_WORDS)
        if ar_score > fa_score:
            return LangGuess("ar", ar_ratio)
        return LangGuess("fa", ar_ratio)
    if ar_ratio <= 0.1:
        return LangGuess("en", 1 - ar_ratio)
    return LangGuess("mixed", max(ar_ratio, 1 - ar_ratio))


class FastTextLangID:
    """Optional wrapper around a fastText / GlotLID model file."""

    def __init__(self, model_path: str):
        try:
            import fasttext  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install fasttext (or fasttext-wheel) to use FastTextLangID") from exc
        self.model = fasttext.load_model(model_path)

    def __call__(self, text: str) -> LangGuess:  # pragma: no cover - needs model file
        labels, probs = self.model.predict(text.replace("\n", " "), k=1)
        label = labels[0].replace("__label__", "")
        code = label.split("_")[0]
        mapping = {"fas": "fa", "pes": "fa", "fa": "fa", "eng": "en", "en": "en", "arb": "ar", "ar": "ar"}
        return LangGuess(mapping.get(code, code), float(probs[0]))
