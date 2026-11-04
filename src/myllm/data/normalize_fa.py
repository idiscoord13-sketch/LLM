"""Persian text normalization (roadmap §4.2 step 1).

Dependency-free so it runs everywhere; if ``hazm`` is installed you can pass
``use_hazm=True`` to additionally run its normalizer.

What it fixes:
  * Arabic letter variants -> Persian (ي/ى -> ی, ك -> ک)
  * Arabic-Indic digits -> Persian digits (or ASCII, configurable)
  * diacritics (harakat) and tatweel removal
  * invisible characters (LRM/RLM/BOM/ZWSP), NBSP -> space
  * ZWNJ (نیم‌فاصله) cleanup + common affixes: «می شود» -> «می‌شود», «کتاب ها» -> «کتاب‌ها»
  * spacing around Persian punctuation and collapsing whitespace
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"

_CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH
    "ۀ": "هٔ",  # HEH WITH YEH ABOVE (ۀ) -> decomposed form, normalized below
    " ": " ",  # NBSP
    " ": " ",
    " ": " ",
}
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "0123456789"

_INVISIBLE = re.compile("[​‎‏‪-‮⁦-⁩﻿]")
_DIACRITICS = re.compile("[ً-ْٰٔ]")  # harakat, superscript alef, hamza above
_TATWEEL = "ـ"

_PERSIAN_LETTER = "؀-ۿﭐ-﷿ﹰ-﻿"
_MI_PREFIX = re.compile(rf"(?<![{_PERSIAN_LETTER}])(ن?می)[ ]+(?=[{_PERSIAN_LETTER}])")
_HA_SUFFIX = re.compile(
    rf"(?<=[{_PERSIAN_LETTER}])[ ]+(ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان)(?![{_PERSIAN_LETTER}])"
)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([،؛؟!.:,;?)\]»])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[«])[ \t]+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_ZWNJ = re.compile(f"{ZWNJ}{{2,}}")
_ZWNJ_NEAR_SPACE = re.compile(f"[ ]*{ZWNJ}[ ]+|[ ]+{ZWNJ}[ ]*")


def to_persian_digits(text: str) -> str:
    return text.translate(str.maketrans(_ARABIC_DIGITS + _ASCII_DIGITS, _PERSIAN_DIGITS * 2))


def to_ascii_digits(text: str) -> str:
    return text.translate(str.maketrans(_ARABIC_DIGITS + _PERSIAN_DIGITS, _ASCII_DIGITS * 2))


def normalize_fa(
    text: str,
    *,
    digits: str = "keep_ascii",
    remove_diacritics: bool = True,
    fix_affixes: bool = True,
    use_hazm: bool = False,
) -> str:
    """Normalize Persian text.

    digits:
        ``keep_ascii`` (default) — Arabic-Indic digits become Persian, ASCII digits stay
        (code, versions and math keep working);
        ``persian`` — every digit becomes Persian; ``ascii`` — every digit becomes ASCII.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(str.maketrans(_CHAR_MAP))
    text = text.replace(_TATWEEL, "")
    text = _INVISIBLE.sub("", text)
    if remove_diacritics:
        text = _DIACRITICS.sub("", text)
    text = unicodedata.normalize("NFC", text)

    if digits == "persian":
        text = to_persian_digits(text)
    elif digits == "ascii":
        text = to_ascii_digits(text)
    else:
        text = text.translate(str.maketrans(_ARABIC_DIGITS, _PERSIAN_DIGITS))

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_ZWNJ.sub(ZWNJ, text)
    text = _ZWNJ_NEAR_SPACE.sub(" ", text)
    if fix_affixes:
        text = _MI_PREFIX.sub(lambda m: m.group(1) + ZWNJ, text)
        text = _HA_SUFFIX.sub(lambda m: ZWNJ + m.group(1), text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text).strip()

    if use_hazm:
        try:
            from hazm import Normalizer  # type: ignore

            text = Normalizer().normalize(text)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("use_hazm=True needs `pip install hazm`") from exc
    return text


def normalize_preserving_code(text: str, **kwargs) -> str:
    """Normalize prose but leave fenced code blocks (```...```) byte-for-byte intact."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    return "".join(p if p.startswith("```") else normalize_fa(p, **kwargs) for p in parts)
