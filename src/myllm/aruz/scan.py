"""Scan a hemistich / couplet: find the metre, the syllable division and the اختیارات used.

    python -m myllm.aruz.scan "بشنو این نی چون شکایت می‌کند" "از جدایی‌ها حکایت می‌کند"

Algorithm
1. every written word part → its plausible readings (``phonology.word_readings``), re-weighted
   by the corpus-learned lexicon when available (``lexicon.py``);
2. beam search left→right over words, adding the cross-word phenomena of verse:
   * حذف همزه (اختیار زبانی): «دل از» → de-laz — a final consonant becomes the onset of a
     following vowel-initial word;
   * کسره‌ی اضافه (unwritten): «دلِ من»; conjunction «و» read -o, va or (اختیار) long u;
   * بلند خواندن مصوت کوتاه پایانی (اختیار زبانی): «که» ke → kē;
3. the hemistich's weight string (last syllable always counts long) is aligned to every
   metre with the اختیارات وزنی: فاعلاتن به جای فعلاتن در آغاز، تسکین (یک بلند به جای دو کوتاه)،
   ابدال (دو کوتاه به جای یک بلند)، قلب (مفاعلن به جای مفتعلن) — precomputed per metre in MeterIndex;
4. reading cost + alignment cost → the best (metre, scansion); the gap to the runner-up
   metre is the confidence.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from myllm.aruz.meters import FEET, METERS, Meter
from myllm.aruz.phonology import Reading, Syl, add_izafe, lengthen_final, split_words, word_readings

BEAM = 300
COST = {
    "hazf_hamze": 0.05, "izafe": 0.5, "lengthen": 0.35, "vav_va": 0.4, "vav_long": 0.35,
    "initial_long": 0.25, "taskin": 0.45, "taskin_final": 0.15, "ebdal": 0.6, "qalb": 0.4,
}
FA_NOTES = {
    "hazf_hamze": "حذف همزه",
    "lengthen": "بلند خواندن مصوت کوتاه پایانی",
    "vav_long": "بلند خواندن «و»",
    "initial_long": "«فاعلاتن» به جای «فعلاتن» در آغاز مصراع",
    "taskin": "تسکین (یک هجای بلند به جای دو کوتاه)",
    "ebdal": "ابدال (دو هجای کوتاه به جای یک بلند)",
    "qalb": "قلب («مفاعلن» به جای «مفتعلن»)",
}


MAX_OPS = 3


class MeterIndex:
    """All text realisations of each metre with <= MAX_OPS اختیارات وزنی, plus their prefixes.

    Turning alignment into a lookup lets the beam prune any reading whose prefix fits no
    metre — the same way a trained reader rejects a scansion half-way through the line.
    """

    def __init__(self, meters: list[Meter], max_ops: int = MAX_OPS):
        self.meters = meters
        self.variants: dict[str, list[tuple[float, int, tuple]]] = {}
        self.prefixes: set[str] = {""}
        for mi, m in enumerate(meters):
            best: dict[str, tuple[float, tuple]] = {}
            self._expand(m, 0, "", 0.0, (), 0, best, max_ops)
            for text, (cost, ops) in best.items():
                self.variants.setdefault(text, []).append((cost, mi, ops))
                for n in range(1, len(text)):
                    self.prefixes.add(text[:n])

    def _expand(self, m: Meter, j: int, text: str, cost: float, ops: tuple, nops: int, best: dict, max_ops: int):
        p = m.pattern
        if j == len(p):
            if text not in best or cost < best[text][0]:
                best[text] = (cost, ops)
            return
        i = len(text)
        steps = [(p[j], 1, 0.0, "=")]
        if nops < max_ops:
            if j == 0 and m.initial_long_ok and p[0] == "u":
                steps.append(("-", 1, COST["initial_long"], "initial_long"))
            if p[j : j + 2] == "uu":
                # «فعلن» -> «فع لن» in the last foot is near-standard; elsewhere تسکین is rarer
                c_taskin = COST["taskin_final"] if j + 2 >= len(p) - 1 else COST["taskin"]
                steps.append(("-", 2, c_taskin, "taskin"))
            if p[j] == "-" and j < len(p) - 1:
                steps.append(("uu", 1, COST["ebdal"], "ebdal"))
            if j in m.qalb_positions and p[j : j + 2] == "-u":
                steps.append(("u-", 2, COST["qalb"], "qalb"))
        for out, adv, c, op in steps:
            extra = 0 if op == "=" else 1
            self._expand(m, j + adv, text + out, cost + c, ops + ((op, i, i + len(out), j, j + adv),),
                         nops + extra, best, max_ops)


_INDEX_CACHE: dict[tuple, MeterIndex] = {}


def meter_index(meters: list[Meter]) -> MeterIndex:
    key = tuple(m.pattern + m.name for m in meters)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = MeterIndex(meters)
    return _INDEX_CACHE[key]


@dataclass
class Hyp:
    syls: tuple[Syl, ...]
    cost: float
    notes: tuple[tuple[str, str], ...] = ()  # (kind, word)
    words: tuple[tuple[str, int, int], ...] = ()  # (word, first_syl, last_syl_exclusive)


@dataclass
class ScanResult:
    text: str
    meter: Meter | None
    cost: float
    syllables: list[str] = field(default_factory=list)
    weights: list[str] = field(default_factory=list)  # per syllable: u, -, -u (last forced long)
    feet_groups: list[tuple[str, list[str]]] = field(default_factory=list)
    ikhtiyarat: list[str] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def ok(self) -> bool:
        return self.meter is not None


class Scanner:
    def __init__(self, lexicon: dict[str, dict[str, float]] | None = None, meters: list[Meter] | None = None,
                 beam: int = BEAM):
        self.lexicon = lexicon or {}
        self.meters = meters or METERS
        self.beam = beam
        self._cache: dict[str, tuple[Reading, ...]] = {}

    # -- readings ---------------------------------------------------------------------------
    def readings(self, part: str) -> tuple[Reading, ...]:
        if part in self._cache:
            return self._cache[part]
        rs = word_readings(part)
        lex = self.lexicon.get(part)
        if lex:
            adj = []
            for r in rs:
                p = lex.get(r.signature)
                bonus = -0.6 * math.log(p) if p else 1.0
                adj.append(Reading(r.syls, round(r.cost * 0.3 + bonus, 3), r.glottal_start, r.tags))
            rs = tuple(sorted(adj, key=lambda r: r.cost))
        self._cache[part] = rs
        return rs

    # -- beam over words ----------------------------------------------------------------------
    def hypotheses(self, text: str) -> list[Hyp]:
        words = split_words(text)
        prefixes = meter_index(self.meters).prefixes
        beam = [Hyp((), 0.0)]
        for w_idx, word_parts in enumerate(words):
            word = "\u200c".join(word_parts)
            final_word = w_idx == len(words) - 1  # no اضافه / lengthening at the end of a hemistich
            for k, part in enumerate(word_parts):
                last_part = k == len(word_parts) - 1 and not final_word
                new: list[Hyp] = []
                for h in beam:
                    if part == "و" and len(word_parts) == 1:
                        new.extend(self._vav(h, word))
                        continue
                    for r in self.readings(part):
                        variants = [r]
                        if last_part:
                            iz = r if "izafe" in r.tags else add_izafe(r, COST["izafe"])
                            if iz is not r:
                                variants.append(iz)
                            for base in (r, iz):  # «سریرِ تو» -> sa-rī-rē: کسره‌ی اضافه بلند خوانده می‌شود
                                lr = lengthen_final(base, COST["lengthen"])
                                if lr is not None:
                                    variants.append(lr)
                        for v in variants:
                            notes = h.notes + (("lengthen", word),) if "lengthen" in v.tags and "lengthen" not in r.tags else h.notes
                            new.append(Hyp(h.syls + v.syls, h.cost + v.cost, notes, h.words))
                            linked = self._link(h, v, word)
                            if linked is not None:
                                new.append(linked)
                new = [h for h in new if _committed(h.syls) in prefixes]
                new.sort(key=lambda x: x.cost)
                beam = _dedupe(new)[: self.beam]
            beam = [Hyp(h.syls, h.cost, h.notes, h.words + ((word, _word_start(h, word), len(h.syls)),)) for h in beam]
        return beam

    @staticmethod
    def _link(h: Hyp, r: Reading, word: str) -> Hyp | None:
        if not h.syls or not r.glottal_start or not h.syls[-1].coda:
            return None
        last = h.syls[-1]
        moved = last.coda[-1]
        new_last = Syl(last.onset, last.long, last.coda[:-1], last.text[:-1] or last.text)
        first = r.syls[0]
        new_first = Syl(moved, first.long, first.coda, moved + first.text[1:])
        syls = h.syls[:-1] + (new_last, new_first) + r.syls[1:]
        return Hyp(syls, h.cost + r.cost + COST["hazf_hamze"], h.notes + (("hazf_hamze", word),), h.words)

    @staticmethod
    def _vav(h: Hyp, word: str) -> list[Hyp]:
        out = []
        if h.syls and h.syls[-1].coda:
            last = h.syls[-1]
            c = last.coda[-1]
            base = h.syls[:-1] + (Syl(last.onset, last.long, last.coda[:-1], last.text[:-1] or last.text),)
            out.append(Hyp(base + (Syl(c, False, (), c + "ُ"),), h.cost, h.notes, h.words))
            out.append(Hyp(base + (Syl(c, True, (), c + "و"),), h.cost + COST["vav_long"],
                           h.notes + (("vav_long", word),), h.words))
        else:
            out.append(Hyp(h.syls + (Syl("و", False, (), "و"),), h.cost + 0.1, h.notes, h.words))
            out.append(Hyp(h.syls + (Syl("و", True, (), "و"),), h.cost + COST["vav_long"],
                           h.notes + (("vav_long", word),), h.words))
        out.append(Hyp(h.syls + (Syl("و", False, (), "وَ"),), h.cost + COST["vav_va"], h.notes, h.words))
        return out

    # -- metre alignment -----------------------------------------------------------------------
    @staticmethod
    def units(syls: tuple[Syl, ...]) -> tuple[str, list[int]]:
        """Weight units (overlong = '-' + 'u'); the hemistich's last syllable counts long."""
        out, owner = [], []
        for k, s in enumerate(syls):
            w = "-" if k == len(syls) - 1 else s.weight
            for ch in w:
                out.append(ch)
                owner.append(k)
        return "".join(out), owner

    # -- public ----------------------------------------------------------------------------------
    def score_meters(self, text: str) -> tuple[dict[str, tuple[float, Hyp, list]], list[Hyp]]:
        hyps = self.hypotheses(text)
        index = meter_index(self.meters)
        best: dict[str, tuple[float, Hyp, list]] = {}
        for h in hyps:
            u, _ = self.units(h.syls)
            for cost, mi, ops in index.variants.get(u, ()):
                m = self.meters[mi]
                total = h.cost + cost
                if m.pattern not in best or total < best[m.pattern][0]:
                    best[m.pattern] = (total, h, list(ops))
        return best, hyps

    def scan(self, text: str, restrict: list[Meter] | None = None) -> ScanResult:
        if restrict is not None:
            saved, self.meters = self.meters, restrict
            try:
                return self.scan(text)
            finally:
                self.meters = saved
        best, _ = self.score_meters(text)
        return self._result(text, best)

    def scan_couplet(self, first: str, second: str) -> tuple[ScanResult, ScanResult]:
        b1, _ = self.score_meters(first)
        b2, _ = self.score_meters(second)
        joint = {p: (b1[p][0] + b2[p][0], p) for p in b1.keys() & b2.keys()}
        if not joint:
            return self._result(first, b1), self._result(second, b2)
        order = sorted(joint.values())
        pick = order[0][1]
        ranking = [(_meter_by_pattern(self.meters, p).name, round(c, 3)) for c, p in order[:5]]
        r1 = self._result(first, {pick: b1[pick]}, ranking, order)
        r2 = self._result(second, {pick: b2[pick]}, ranking, order)
        return r1, r2

    def _result(self, text, best, ranking=None, order=None) -> ScanResult:
        if not best:
            return ScanResult(text, None, float("inf"))
        items = sorted(best.items(), key=lambda kv: kv[1][0])
        pattern, (cost, hyp, ops) = items[0]
        meter = _meter_by_pattern(self.meters, pattern)
        if ranking is None:
            ranking = [(_meter_by_pattern(self.meters, p).name, round(v[0], 3)) for p, v in items[:5]]
        costs = [c for _, c in ranking]
        gap = (costs[1] - costs[0]) if len(costs) > 1 else 2.0
        units, owner = self.units(hyp.syls)
        sylls = [s.text or "؟" for s in hyp.syls]
        weights = [("-" if k == len(hyp.syls) - 1 else s.weight) for k, s in enumerate(hyp.syls)]
        notes = [f"{FA_NOTES[k]} («{w}»)" for k, w in hyp.notes if k in FA_NOTES]
        notes += [FA_NOTES[op] for op, *_ in ops if op in FA_NOTES]
        return ScanResult(text, meter, round(cost, 3), sylls, weights, _feet_groups(meter, ops, owner, sylls),
                          notes, ranking, round(1 - math.exp(-2 * gap), 3))


def _committed(syls: tuple[Syl, ...]) -> str:
    """Weight units of all but the last syllable (which linking/final rules may still change)."""
    return "".join(s.weight for s in syls[:-1])


def _word_start(h: Hyp, word: str) -> int:
    return h.words[-1][2] if h.words else 0


def _dedupe(hyps: list[Hyp]) -> list[Hyp]:
    seen, out = set(), []
    for h in hyps:
        key = tuple((s.weight, s.coda) for s in h.syls[-2:]) + (Scanner.units(h.syls)[0],)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _meter_by_pattern(meters: list[Meter], pattern: str) -> Meter:
    return next(m for m in meters if m.pattern == pattern)


def _feet_groups(meter: Meter, ops, owner: list[int], sylls: list[str]) -> list[tuple[str, list[str]]]:
    """Group syllables under the metre's feet (رکن‌ها) using the alignment."""
    bounds, pos = [], 0
    for f in meter.feet:
        pos += len(FEET[f])
        bounds.append(pos)
    groups: list[tuple[str, list[int]]] = [(f, []) for f in meter.feet]
    for _op, ti, tj, pj, _pj2 in ops:
        foot = next(n for n, b in enumerate(bounds) if pj < b)
        for t in range(ti, tj):
            s = owner[t]
            if s not in groups[foot][1] and all(s not in g[1] for g in groups):
                groups[foot][1].append(s)
    return [(f, [sylls[s] for s in idx]) for f, idx in groups]


def format_result(r: ScanResult) -> str:
    if not r.ok:
        return f"«{r.text}»\nوزنی پیدا نشد (ممکن است بیت عربی، نیمایی یا با واژه‌ی ناآشنا باشد)."
    marks = " ".join("–" if w == "-" else ("–◡" if w == "-u" else "◡") for w in r.weights)
    lines = [
        f"«{r.text}»",
        f"وزن: {r.meter.feet_label()} — {r.meter.name}",
        "تقطیع: " + " | ".join(r.syllables),
        "نشانه‌ها: " + marks,
        "ارکان: " + "  ".join(f"[{' '.join(s)}] {f}" for f, s in r.feet_groups),
    ]
    if r.ikhtiyarat:
        lines.append("اختیارات: " + "، ".join(dict.fromkeys(r.ikhtiyarat)))
    level = "بالا" if r.confidence >= 0.7 else "متوسط" if r.confidence >= 0.4 else "پایین"
    lines.append(f"اطمینان: {level}")
    if r.confidence < 0.7 and len(r.ranking) > 1:
        lines.append("گزینه‌ی بعدی: " + r.ranking[1][0])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="تقطیع و تشخیص وزن")
    p.add_argument("hemistichs", nargs="+", help="one hemistich, or two for a couplet")
    p.add_argument("--lexicon", help="learned lexicon JSON (myllm.aruz.lexicon)")
    a = p.parse_args(argv)
    from myllm.aruz.lexicon import load_lexicon

    sc = Scanner(load_lexicon(a.lexicon) if a.lexicon else None)
    if len(a.hemistichs) >= 2:
        for r in sc.scan_couplet(a.hemistichs[0], a.hemistichs[1]):
            print(format_result(r), end="\n\n")
    else:
        print(format_result(sc.scan(a.hemistichs[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
