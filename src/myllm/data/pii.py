"""PII scrubbing with Iranian formats (roadmap §4.2 step 5, §7.3 step 2).

Regex + checksum validation, so random 10/16-digit numbers are not destroyed:
  * email, IPv4, URLs with credentials
  * Iranian mobile (09xx / +989xx / 00989xx) and landline (0xx-xxxxxxxx)
  * national code (کد ملی) — validated with the official mod-11 checksum
  * bank card — validated with Luhn
  * IBAN / شبا (IR + 24 digits) — validated with ISO 13616 mod-97

Persian and Arabic-Indic digits are matched too (they are mapped 1:1 to ASCII for
matching, so character offsets stay aligned).

Names and street addresses need an NER model; plug one in through ``extra_scrubbers``.
Regex PII removal is a floor, not a guarantee — keep raw user data encrypted and separate.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_CREDS = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@[^\s]+", re.IGNORECASE)
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
MOBILE = re.compile(r"(?<!\d)(?:\+98|0098|0)?[ -]?9\d{2}[ -]?\d{3}[ -]?\d{4}(?!\d)")
LANDLINE = re.compile(r"(?<!\d)0[1-8]\d[ -]?\d{8}(?!\d)")
IBAN = re.compile(r"\bIR[ ]?\d{2}(?:[ ]?\d{4}){5}[ ]?\d{2}\b", re.IGNORECASE)
CARD = re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")
NATIONAL_ID = re.compile(r"(?<!\d)\d{3}[ -]?\d{6}[ -]?\d(?!\d)")


def valid_national_id(code: str) -> bool:
    digits = re.sub(r"\D", "", code.translate(_DIGIT_MAP))
    if len(digits) != 10 or len(set(digits)) == 1:
        return False
    check = int(digits[9])
    s = sum(int(digits[i]) * (10 - i) for i in range(9)) % 11
    return check == s if s < 2 else check == 11 - s


def valid_luhn(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number.translate(_DIGIT_MAP))]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def valid_iban(iban: str) -> bool:
    s = re.sub(r"\s", "", iban.translate(_DIGIT_MAP)).upper()
    if not re.fullmatch(r"IR\d{24}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


# (label, pattern, validator) — order matters: longer / more specific first.
_RULES: list[tuple[str, re.Pattern, Callable[[str], bool] | None]] = [
    ("URL_CREDENTIALS", URL_CREDS, None),
    ("EMAIL", EMAIL, None),
    ("IBAN", IBAN, valid_iban),
    ("CARD", CARD, valid_luhn),
    ("NATIONAL_ID", NATIONAL_ID, valid_national_id),
    ("PHONE", MOBILE, None),
    ("PHONE", LANDLINE, None),
    ("IP", IPV4, None),
]


def scrub_pii(
    text: str,
    *,
    extra_scrubbers: list[Callable[[str], tuple[str, Counter]]] | None = None,
) -> tuple[str, Counter]:
    """Replace PII with ``[LABEL]`` placeholders. Returns (clean_text, counts per label)."""
    counts: Counter = Counter()
    if not text:
        return text, counts
    for label, pattern, validator in _RULES:
        matchable = text.translate(_DIGIT_MAP)  # same length as text -> offsets align
        out: list[str] = []
        last = 0
        for m in pattern.finditer(matchable):
            span_text = matchable[m.start() : m.end()]
            if validator is not None and not validator(span_text):
                continue
            out.append(text[last : m.start()])
            out.append(f"[{label}]")
            counts[label] += 1
            last = m.end()
        out.append(text[last:])
        text = "".join(out)
    for scrubber in extra_scrubbers or []:
        text, extra = scrubber(text)
        counts.update(extra)
    return text, counts


def contains_pii(text: str) -> bool:
    return sum(scrub_pii(text)[1].values()) > 0
