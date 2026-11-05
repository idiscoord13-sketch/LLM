"""Persian syllabification for prosody (تقطیع هجایی) — from script to syllable weights.

The hard part of scanning Persian verse is that short vowels (a, e, o) are not written.
For metre we do not need *which* short vowel it is — only *whether* a written consonant is
followed by one. So for every word we enumerate the phonotactically valid readings:

  * a syllable is always  C V (C)(C)  — exactly one onset consonant, at most two in the coda
  * ا (word-initial) / آ carry a glottal onset ʔ; mid-word ا is the long vowel ā
  * و / ی can be a consonant (v, y), a long vowel (u, i), and و can be short o (تو، دو، خود)
  * word-final ه after a consonant is usually silent and marks a short vowel (نامه، که)
  * harakat, when present (Ganjoor vowelises many verses), force or forbid short vowels;
    tashdid doubles a consonant; tanvin adds "-an"

Each reading gets a prior cost (fewer inserted vowels, no odd clusters, ...); the metre
search in ``scan.py`` and the corpus-learned lexicon (``lexicon.py``) pick among them.

Syllable weights (textbook علوم و فنون ادبی):
  CV → کوتاه (u) · CVC, CVː → بلند (-) · CVːC, CVCC, CVːCC → کشیده (-u)
  and «ن» after a long vowel is not counted (جان، این، چون → long, not overlong).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
SUKUN = {"ْ", "ۡ"}
SHADDA = "ّ"
TANVIN = {"ً": "a", "ٍ": "e", "ٌ": "o"}
SUPERSCRIPT_ALEF = "ٰ"
HAMZA_ABOVE = "ٔ"
SHORT_MARKS = {FATHA: "a", KASRA: "e", DAMMA: "o"}
MARKS = set(SHORT_MARKS) | SUKUN | {SHADDA, SUPERSCRIPT_ALEF, HAMZA_ABOVE} | set(TANVIN)
ZWNJ = "‌"
GLOTTAL = "ʔ"

_LETTER_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه" + HAMZA_ABOVE, "ة": "ه", "ٱ": "ا", "‍": ""})
_PUNCT = re.compile(r"[،؛؟!«»:\"'.,;?()\[\]\-–—…ـ]+")
VOWEL_LETTERS = {"ا", "آ", "و", "ی"}
HAMZA_LETTERS = {"ء", "أ", "إ", "ؤ", "ئ"}


def normalize_verse(text: str) -> str:
    """Light normalisation that KEEPS harakat (unlike myllm.data.normalize_fa)."""
    text = unicodedata.normalize("NFC", text).translate(_LETTER_FIX).replace(" ", " ")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Syl:
    onset: str
    long: bool  # long vowel (ā, i, u) vs short (a, e, o)
    coda: tuple[str, ...]
    text: str  # written letters (for display)

    @property
    def weight(self) -> str:
        if not self.long:
            return "u" if not self.coda else ("-" if len(self.coda) == 1 else "-u")
        if not self.coda or self.coda == ("ن",):
            return "-"
        return "-u"


@dataclass(frozen=True)
class Reading:
    syls: tuple[Syl, ...]
    cost: float
    glottal_start: bool = False  # starts with ʔ from ا/آ -> eligible for حذف همزه
    tags: tuple[str, ...] = ()

    @property
    def weights(self) -> str:
        return "".join(s.weight for s in self.syls)

    @property
    def signature(self) -> str:
        """Compact key used by the lexicon: weights + syllable shapes."""
        return ".".join(("A" if s.long else "V") + str(len(s.coda)) for s in self.syls)


def _split_letters(word: str) -> list[tuple[str, frozenset[str]]]:
    out: list[tuple[str, set[str]]] = []
    for ch in word:
        if ch in MARKS:
            if out:
                out[-1][1].add(ch)
        else:
            out.append((ch, set()))
    return [(ch, frozenset(m)) for ch, m in out]


# --- phoneme alternatives per letter -------------------------------------------------------
# phoneme items: ("C", letter, vowel_rule) with vowel_rule in {"free", "forced", "none"}, or ("A", letter)
def _alternatives(letters, i: int) -> list[tuple[list, float]]:
    ch, marks = letters[i]
    first, last = i == 0, i == len(letters) - 1
    prev = letters[i - 1][0] if i else None
    # word-initial ا is a consonant (glottal onset): «این» ʔ-i-n, «او» ʔ-u
    prev_consonantal = i > 0 and (prev not in VOWEL_LETTERS or (prev == "ا" and i == 1))
    nxt = letters[i + 1][0] if not last else None
    vowel_rule = "forced" if marks & set(SHORT_MARKS) else ("none" if marks & SUKUN else "free")
    gem = SHADDA in marks

    def cons(letter=ch, rule=vowel_rule):
        items = [("C", letter, "none")] if gem else []
        return items + [("C", letter, rule)]

    if marks & set(TANVIN):  # کأساً -> ka'-san: seat letter carries "-an"
        base = [] if ch == "ا" else cons(rule="none")
        return [(base + [("V", ch), ("C", "ن", "none")], 0.0)]
    if ch == "آ":
        alts = [([("C", GLOTTAL, "none"), ("A", "آ")], 0.0)]
        if not first:
            alts.append(([("A", "آ")], 0.2))  # کآخر -> kā-xar
        return alts
    if ch == "ا":
        if first:
            return [([("C", GLOTTAL, vowel_rule)], 0.0)]
        alts = [([("A", "ا")], 0.0)]
        if prev not in VOWEL_LETTERS and nxt is not None and nxt not in VOWEL_LETTERS:
            # prefix + vowel-initial stem: «برافشانیم» bar+afšān -> ba-raf-šā-nīm, «دراندازیم» da-ran-dā-zīm
            alts.append(([("V", "ا")], 0.45))
        return alts
    if ch == "و":
        if prev == "خ" and nxt in ("ا", "ی"):
            return [([], 0.0)]  # و معدوله: خواب، خویش
        alts = []
        alts.append(([("A", "و")], 0.0 if prev_consonantal else 0.5))
        alts.append((cons(), 0.45 if prev_consonantal else 0.0))
        if last and prev_consonantal:
            alts.append(([("V", "و")], 0.15 if i == 1 else 0.25))  # تو، دو، چو؛ بشنو، برو
        elif prev == "خ":
            alts.append(([("V", "و")], 0.15))  # خود، خوش
        return alts
    if ch == "ی":
        if SUPERSCRIPT_ALEF in marks:
            return [([("A", "ی")], 0.0)]  # متیٰ
        return [([("A", "ی")], 0.0 if prev_consonantal else 0.4), (cons(), 0.45 if prev_consonantal else 0.0)]
    if ch == "ه" and last and not first:
        if prev in ("ا", "و", "آ"):
            return [(cons(), 0.0)]  # ماه، شاه، کوه
        alts = [([("V", "ه")], 0.0), (cons(), 0.6)]  # silent ه (نامه، که) vs consonant h (ده، به)
        return alts
    if ch in HAMZA_LETTERS:
        return [(cons(GLOTTAL), 0.0)]
    return [(cons(), 0.0)]


_INSERT_COST = 0.12
_CLUSTER_INTERNAL = 0.4
_CLUSTER_FINAL = 0.1


def _syllabify(phonemes: list[tuple]) -> tuple[list[Syl], float] | None:
    """Phoneme list -> syllables (or None if phonotactically invalid)."""
    vowel_idx = [k for k, p in enumerate(phonemes) if p[0] in ("V", "A")]
    if not vowel_idx or phonemes[0][0] != "C" or vowel_idx[0] != 1:
        return None
    syls, extra = [], 0.0
    for n, vi in enumerate(vowel_idx):
        onset = phonemes[vi - 1]
        if onset[0] != "C":
            return None
        nxt_v = vowel_idx[n + 1] if n + 1 < len(vowel_idx) else None
        end = (nxt_v - 1) if nxt_v is not None else len(phonemes)
        coda = phonemes[vi + 1 : end]
        if any(p[0] != "C" for p in coda) or len(coda) > 2:
            return None
        if nxt_v is not None and nxt_v - vi < 2:
            return None  # vowel hiatus without onset
        v = phonemes[vi]
        if len(coda) == 2:
            extra += _CLUSTER_FINAL if nxt_v is None else _CLUSTER_INTERNAL
            if v[0] == "A":
                extra += 0.25  # long vowel + 2 consonants (داشت، دوست) is rarer than CVCC
        letters = [p[1] for p in [onset, v, *coda]]
        if onset[1] == GLOTTAL and v[1] in ("آ", "ا"):
            letters = letters[1:]  # «آ» already shows the glottal onset
        text = "".join(letters).replace(GLOTTAL, "ا")
        syls.append(Syl(onset[1], v[0] == "A", tuple(p[1] for p in coda), text))
    return syls, extra


@lru_cache(maxsize=200_000)
def word_readings(word: str, max_readings: int = 48) -> tuple[Reading, ...]:
    """All plausible readings of one written word (part between spaces / ZWNJ)."""
    letters = _split_letters(word)
    if not letters:
        return ()
    explicit_izafe = letters[-1][0] == "ه" and HAMZA_ABOVE in letters[-1][1]
    if explicit_izafe:
        letters[-1] = ("ه", letters[-1][1] - {HAMZA_ABOVE})
    final_kasra = KASRA in letters[-1][1]
    alts = [_alternatives(letters, i) for i in range(len(letters))]
    results: dict[str, Reading] = {}

    def dfs(i: int, phon: list, cost: float):
        if cost > 2.5:
            return
        if i == len(letters):
            parsed = _syllabify(phon)
            if parsed is None:
                return
            syls, extra = parsed
            r = Reading(tuple(syls), round(cost + extra, 3), glottal_start=phon[0][1] == GLOTTAL)
            key = r.signature
            if key not in results or results[key].cost > r.cost:
                results[key] = r
            return
        for items, c in alts[i]:
            expansions = [(list(phon), cost + c)]
            for it in items:
                new = []
                for ph, cc in expansions:
                    if it[0] == "C":
                        rule = it[2]
                        if rule in ("free", "none"):
                            new.append((ph + [it], cc))
                        if rule == "forced":
                            new.append((ph + [it, ("V", "")], cc))
                        elif rule == "free":
                            new.append((ph + [it, ("V", "")], cc + _INSERT_COST))
                    else:
                        new.append((ph + [it], cc))
                expansions = new
            for ph, cc in expansions:
                if _prefix_ok(ph):
                    dfs(i + 1, ph, cc)

    dfs(0, [], 0.0)
    readings = sorted(results.values(), key=lambda r: r.cost)[:max_readings]
    if explicit_izafe:
        readings = [add_izafe(r, cost=0.0) for r in readings]
    elif final_kasra:
        readings = [Reading(r.syls, r.cost, r.glottal_start, ("izafe",)) for r in readings]
    return tuple(readings)


def _prefix_ok(ph: list) -> bool:
    if not ph:
        return True
    if ph[0][0] != "C":
        return False
    run = 0
    prev_vowel = False
    for p in ph:
        if p[0] == "C":
            run += 1
            prev_vowel = False
            if run > 3:
                return False
        else:
            if prev_vowel or run == 0:
                return False
            run = 0
            prev_vowel = True
    return True


def add_izafe(r: Reading, cost: float = 0.5) -> Reading:
    """Unwritten کسره‌ی اضافه: «دل» del → «دلِ» de-le; «پا» pā → «پای» pā-ye."""
    syls = list(r.syls)
    last = syls[-1]
    if last.coda:
        moved = last.coda[-1]
        syls[-1] = Syl(last.onset, last.long, last.coda[:-1], last.text[:-1] if len(last.text) > 1 else last.text)
        syls.append(Syl(moved, False, (), moved + "ِ"))
    else:
        syls.append(Syl("ی", False, (), "یِ"))
    return Reading(tuple(syls), r.cost + cost, r.glottal_start, r.tags + ("izafe",))


def lengthen_final(r: Reading, cost: float = 0.35) -> Reading | None:
    """اختیار زبانی: final short open vowel read long (که ke → kē, تو to → tū)."""
    last = r.syls[-1]
    if last.long or last.coda:
        return None
    syls = list(r.syls[:-1]) + [Syl(last.onset, True, (), last.text)]
    return Reading(tuple(syls), r.cost + cost, r.glottal_start, r.tags + ("lengthen",))


# Vowel-less enclitics written after a ZWNJ belong to the previous syllable, not to a new word:
# «ایوانی‌ست» = ey-vā-nīst, «بی‌خبر‌م» = bī-kha-ba-ram, «گفت‌ش» = gof-taš.
_BOUND_ENCLITICS = {"ست", "م", "ت", "ش", "ند", "یم", "ید"}


def _strip_marks(s: str) -> str:
    return "".join(ch for ch in s if ch not in MARKS)


def split_words(verse: str) -> list[list[str]]:
    """Verse -> words, each a list of ZWNJ-separated parts (می‌رود -> ['می', 'رود'])."""
    words = []
    for w in normalize_verse(verse).split(" "):
        parts: list[str] = []
        for p in w.split(ZWNJ):
            if not p:
                continue
            if parts and _strip_marks(p) in _BOUND_ENCLITICS:
                parts[-1] += p
            else:
                parts.append(p)
        if parts:
            words.append(parts)
    return words
