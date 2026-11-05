"""Student mistake diagnosis: compare a student's scansion with the engine's and explain each error.

    from myllm.aruz.diagnose import diagnose, format_diagnosis
    d = diagnose(scanner, "بشنو این نی چون شکایت می‌کند", "– U – U – U – – – U –")
    print(format_diagnosis(d))

Input signs (any mix): long = «–», «-», «—», «ـ»; short = «U», «u», «◡», «∪». A کشیده (overlong)
syllable is written as two signs «– U», as in the school book. The engine's answer uses the same
convention, the last syllable of a hemistich counted long.

Each disagreement is mapped back to the engine's syllable and explained with the rule a student
most likely missed (نون پس از مصوت بلند، هجای کشیده، ه غیرملفوظ، و معدوله، حذف همزه، کسره‌ی اضافه،
هجای پایانی، بلندخوانی مصوت ...). If the engine itself is unsure, the report says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from myllm.aruz.scan import Scanner, ScanResult

_LONG = set("-–—ـ")
_SHORT = set("uU◡∪ᴗ")


def parse_signs(text: str) -> str:
    """'– U – –' -> '-u--' (anything else is ignored)."""
    out = []
    for ch in text:
        if ch in _LONG:
            out.append("-")
        elif ch in _SHORT:
            out.append("u")
    return "".join(out)


@dataclass
class Mistake:
    position: int  # 1-based sign position in the engine's answer
    syllable: str
    expected: str
    got: str
    reason: str


@dataclass
class Diagnosis:
    text: str
    expected: str
    student: str
    result: ScanResult
    mistakes: list[Mistake] = field(default_factory=list)
    correct: int = 0

    @property
    def score(self) -> float:
        return self.correct / max(1, len(self.expected))


def _align(a: str, b: str) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment of two sign strings -> pairs of indices (None = gap)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


_IZAFE = re.compile("ِ$")


def _reason(scanner: Scanner, r: ScanResult, syl_idx: int, unit_in_syl: int, expected: str, got: str) -> str:
    syl = scanner._last_hyp.syls[syl_idx]
    text = r.syllables[syl_idx]
    last = syl_idx == len(r.syllables) - 1
    word = next((w for w, a, b in scanner._last_hyp.words if a <= syl_idx < b), "")
    notes = {k for k, w in scanner._last_hyp.notes if w == word}
    if last:
        return "هجای آخر مصراع همیشه بلند (–) حساب می‌شود، هر شکلی که داشته باشد."
    if "hazf_hamze" in notes:
        return f"در «{word}» همزه حذف شده (حذف همزه): صامتِ پیش از آن به هجای بعد می‌چسبد و هجاها جابه‌جا می‌شوند."
    if syl.weight == "-u":
        return (f"«{text}» هجای کشیده است (مصوت بلند + صامت، یا مصوت کوتاه + دو صامت) و در تقطیع "
                "«– U» نوشته می‌شود: یک بلند و یک کوتاه.")
    if syl.long and syl.coda == ("ن",):
        return (f"«{text}» مصوت بلند دارد، پس بلند (–) است؛ و «ن» بعد از مصوت بلند در تقطیع حساب نمی‌شود، "
                "پس کشیده («– U») هم نیست.")
    if ("lengthen" in notes or "vav_long" in notes) and syl.long:
        if text.endswith("ِ"):
            return f"کسره‌ی اضافه‌ی بعد از «{word}» اینجا به اختیار شاعری بلند خوانده می‌شود (–)."
        return f"اینجا مصوت کوتاه پایانی «{word}» به اختیار شاعری بلند خوانده می‌شود (–)."
    if _IZAFE.search(text) or text.endswith("یِ"):
        return f"کسره‌ی اضافه بعد از «{word}» خودش یک هجای کوتاه (U) می‌سازد."
    if "خوا" in word or "خوی" in word:
        return f"«و» در «{word}» معدوله است و خوانده نمی‌شود («خا»)."
    if not syl.long and not syl.coda:
        if text.endswith("ه"):
            return f"«ه» پایانی در «{word}» خوانده نمی‌شود و فقط مصوت کوتاه را نشان می‌دهد؛ پس «{text}» کوتاه (U) است."
        return f"«{text}» یک صامت + مصوت کوتاه است (CV) ← هجای کوتاه (U)."
    if syl.long:
        return f"«{text}» مصوت بلند دارد (ا، و، ی) ← هجای بلند (–)."
    return f"«{text}» صامت + مصوت کوتاه + صامت است (CVC) ← هجای بلند (–)."


def diagnose(scanner: Scanner, text: str, student_signs: str) -> Diagnosis:
    r = scanner.scan(text)
    if not r.ok:
        return Diagnosis(text, "", parse_signs(student_signs), r)
    hyp = _best_hyp(scanner, text, r)
    scanner._last_hyp = hyp
    units, owner = scanner.units(hyp.syls)
    student = parse_signs(student_signs)
    d = Diagnosis(text, units, student, r)
    for i, j in _align(units, student):
        if i is not None and j is not None and units[i] == student[j]:
            d.correct += 1
            continue
        if i is None:
            d.mistakes.append(Mistake(j + 1, "", "", student[j], "این نشانه اضافه است: تعداد هجاهای شما از تقطیع درست بیشتر شده."))
            continue
        s_idx = owner[i]
        unit_in_syl = i - owner.index(s_idx)
        got = student[j] if j is not None else ""
        reason = _reason(scanner, r, s_idx, unit_in_syl, units[i], got)
        if j is None:
            reason = "یک نشانه جا افتاده. " + reason
        d.mistakes.append(Mistake(i + 1, r.syllables[s_idx], units[i], got, reason))
    return d


def _best_hyp(scanner: Scanner, text: str, r: ScanResult):
    best, _ = scanner.score_meters(text)
    pattern = r.meter.pattern
    return best[pattern][1]


def _signs(s: str) -> str:
    return " ".join("–" if c == "-" else "U" for c in s)


def format_diagnosis(d: Diagnosis) -> str:
    if not d.expected:
        return "وزن این مصراع را پیدا نکردم، پس نمی‌توانم تقطیع شما را بررسی کنم."
    lines = [f"«{d.text}»", f"وزن: {d.result.meter.feet_label()} — {d.result.meter.name}"]
    if not d.mistakes:
        lines.append(f"✅ آفرین! تقطیع شما کاملاً درست است ({len(d.expected)} نشانه).")
    else:
        extra = sum(1 for m in d.mistakes if not m.expected)
        missing = sum(1 for m in d.mistakes if m.expected and not m.got)
        wrong = len(d.mistakes) - extra - missing
        parts = [f"{d.correct} از {len(d.expected)} نشانه درست"]
        if wrong:
            parts.append(f"{wrong} نشانه‌ی غلط")
        if missing:
            parts.append(f"{missing} نشانه‌ی جاافتاده")
        if extra:
            parts.append(f"{extra} نشانه‌ی اضافه")
        lines.append("❌ " + "، ".join(parts) + ".")
        sign = {"-": "–", "u": "U"}
        for m in d.mistakes[:8]:
            if not m.expected:
                lines.append(f"• نشانه‌ی اضافه («{sign.get(m.got, m.got)}» در جای {m.position} تقطیع شما): {m.reason}")
                continue
            where = f"نشانه‌ی {m.position}" + (f" (هجای «{m.syllable}»)" if m.syllable else "")
            got = f"شما «{sign[m.got]}» نوشتید" if m.got else "شما ننوشتید"
            lines.append(f"• {where}: درستش «{sign[m.expected]}» است، {got}. {m.reason}")
    lines.append("تقطیع درست: " + " | ".join(d.result.syllables))
    lines.append("نشانه‌ها: " + _signs(d.expected))
    if d.result.confidence < 0.7:
        lines.append("⚠️ موتور هم درباره‌ی این مصراع کاملاً مطمئن نیست؛ با معلم هم چک کن.")
    return "\n".join(lines)
