"""Persian metres (بحور عروضی): feet → syllable patterns, plus the metre table.

Patterns use ``u`` for a short (کوتاه) and ``-`` for a long (بلند) syllable, in reading
order (first foot first). Foot names follow the Iranian school book / Ganjoor labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FEET: dict[str, str] = {
    "فعولن": "u--", "فعول": "u-u", "فعل": "u-", "فع": "-", "فاع": "-u", "فعلن": "uu-", "فع لن": "--",
    "مفاعیلن": "u---", "مفاعیل": "u--u", "مفاعلن": "u-u-", "مفاعلتن": "u-uu-",
    "فاعلاتن": "-u--", "فاعلات": "-u-u", "فعلاتن": "uu--", "فعلات": "uu-u", "فاعلن": "-u-",
    "مفعول": "--u", "مفعولن": "---", "مستفعلن": "--u-", "مستفعل": "--u", "مفتعلن": "-uu-", "مفتعلات": "-uu-u",
    "متفاعلن": "uu-u-", "فاعلان": "-u-", "فعلان": "uu-", "فعولان": "u--", "مفاعیلان": "u---",
}
# longest names first so "مفاعیلن" is not parsed as "مفاعیل" + "ن"
_ORDER = sorted(FEET, key=len, reverse=True)


def feet_pattern(feet: str) -> tuple[list[str], str]:
    """'مفعول فاعلات مفاعیل فاعلن' -> (['مفعول', ...], '--u-u-uu--uu-u-')."""
    text = feet.replace("فع لن", "فع‌لن").strip()
    tokens, pattern = [], ""
    for tok in text.split():
        tok = tok.replace("فع‌لن", "فع لن").strip("،,")
        if tok not in FEET:
            raise ValueError(f"unknown foot {tok!r} in {feet!r}")
        tokens.append(tok)
        pattern += FEET[tok]
    return tokens, pattern


@dataclass(frozen=True)
class Meter:
    name: str
    feet: tuple[str, ...]
    pattern: str
    family: str
    aliases: tuple[str, ...] = field(default=())

    @property
    def initial_long_ok(self) -> bool:
        """«فاعلاتن» به جای «فعلاتن» در آغاز مصراع (اختیار وزنی)."""
        return self.feet[0] == "فعلاتن"

    @property
    def qalb_positions(self) -> tuple[int, ...]:
        """Start offsets of مفتعلن feet, where «مفاعلن» may stand instead (قلب)."""
        pos, out = 0, []
        for f in self.feet:
            if f == "مفتعلن":
                out.append(pos)
            pos += len(FEET[f])
        return tuple(out)

    def feet_label(self) -> str:
        return " ".join(self.feet)


def _m(feet: str, name: str, family: str | None = None, aliases: tuple[str, ...] = ()) -> Meter:
    toks, pat = feet_pattern(feet)
    return Meter(name, tuple(toks), pat, family or name.split()[0], aliases)


# Metres seen in Ganjoor's labels (counts in docs/product/aruz.md) + rubāʿī variants.
METERS: list[Meter] = [
    _m("فعولن فعولن فعولن فعل", "متقارب مثمن محذوف", aliases=("وزن شاهنامه",)),
    _m("فعولن فعولن فعولن فعولن", "متقارب مثمن سالم"),
    _m("مفاعیلن مفاعیلن فعولن", "هزج مسدس محذوف", aliases=("وزن دوبیتی",)),
    _m("مفاعیلن مفاعیلن مفاعیلن مفاعیلن", "هزج مثمن سالم"),
    _m("مفعول مفاعلن فعولن", "هزج مسدس اخرب مقبوض محذوف"),
    _m("مفعول مفاعلن مفاعیلن", "هزج مسدس اخرب مقبوض"),
    _m("مفعول مفاعیل مفاعیل فعولن", "هزج مثمن اخرب مکفوف محذوف"),
    _m("مفعول مفاعیلن مفعول مفاعیلن", "هزج مثمن اخرب"),
    _m("مفعول مفاعیل فاعلاتن", "مفعول مفاعیل فاعلاتن (نام بحر در گنجور نیامده)", family="هزج"),
    _m("فعلاتن مفاعلن فعلن", "خفیف مسدس مخبون"),
    _m("فاعلاتن فاعلاتن فاعلن", "رمل مسدس محذوف", aliases=("وزن مثنوی",)),
    _m("فاعلاتن فاعلاتن فاعلاتن فاعلن", "رمل مثمن محذوف"),
    _m("فاعلاتن فاعلاتن فاعلاتن فاعلاتن", "رمل مثمن سالم"),
    _m("فعلاتن فعلاتن فعلاتن فعلن", "رمل مثمن مخبون محذوف"),
    _m("فعلاتن فعلاتن فعلاتن فعلاتن", "رمل مثمن مخبون"),
    _m("فعلاتن فعلاتن فعلاتن فع", "رمل مثمن مخبون (با «فع» پایانی؛ نام در گنجور نیامده)", family="رمل"),
    _m("فعلاتن فعلاتن فعلن", "رمل مسدس مخبون محذوف"),
    _m("فعلات فاعلاتن فعلات فاعلاتن", "رمل مثمن مشکول"),
    _m("مفاعلن فعلاتن مفاعلن فعلن", "مجتث مثمن مخبون محذوف"),
    _m("مفاعلن فعلاتن مفاعلن فعلاتن", "مجتث مثمن مخبون"),
    _m("مفعول فاعلات مفاعیل فاعلن", "مضارع مثمن اخرب مکفوف محذوف"),
    _m("مفعول فاعلاتن مفعول فاعلاتن", "مضارع مثمن اخرب"),
    _m("مفعول فاعلات مفاعیلن", "مضارع مسدس اخرب مکفوف"),
    _m("مستفعلن مستفعلن مستفعلن مستفعلن", "رجز مثمن سالم"),
    _m("مفتعلن مفاعلن مفتعلن مفاعلن", "رجز مثمن مطوی مخبون"),
    _m("مفتعلن مفتعلن فاعلن", "سریع مطوی مکشوف"),
    _m("مفتعلن فاعلات مفتعلن فع", "منسرح مثمن مطوی منحور"),
    _m("مفتعلن فاعلن مفتعلن فاعلن", "منسرح مطوی مکشوف"),
    # rubāʿī (24 traditional variants; the common ones — تسکین/ابدال cover more)
    _m("مفعول مفاعیل مفاعیل فعل", "رباعی (اخرب)", family="رباعی"),
    _m("مفعول مفاعیل مفاعیلن فع", "رباعی (اخرب)", family="رباعی"),
    _m("مفعول مفاعلن مفاعیلن فع", "رباعی (اخرب)", family="رباعی"),
    _m("مفعولن مفعول مفاعیلن فع", "رباعی (اخرم)", family="رباعی"),
    _m("مفعولن مفاعلن مفاعیلن فع", "رباعی (اخرم)", family="رباعی"),
    _m("مفعولن مفعولن مفعولن فع", "رباعی (اخرم)", family="رباعی"),
]

BY_PATTERN: dict[str, Meter] = {}
for _meter in METERS:
    BY_PATTERN.setdefault(_meter.pattern, _meter)


def meter_for_rhythm(rhythm: str) -> Meter | None:
    """Map a Ganjoor ``Metre.Rhythm`` label to a Meter (by pattern; rubāʿī label -> family)."""
    from myllm.aruz.ganjoor import split_rhythm

    feet, name = split_rhythm(rhythm)
    try:
        _, pat = feet_pattern(feet)
    except ValueError:
        return None
    if name and "رباعی" in name:
        return next(m for m in METERS if m.family == "رباعی")
    return BY_PATTERN.get(pat)


def meter_key(m: Meter) -> str:
    """Identity used for evaluation, priors and the classifier: the pattern, or "رباعی" for the whole family."""
    return "رباعی" if m.family == "رباعی" else m.pattern


def same_meter(a: Meter | None, b: Meter | None) -> bool:
    if a is None or b is None:
        return False
    if a.family == "رباعی" and b.family == "رباعی":
        return True
    return a.pattern == b.pattern
