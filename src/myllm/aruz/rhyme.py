"""Rhyme (قافیه) and refrain (ردیف) detection.

* ردیف = the identical word(s) repeated at the end of every rhyming hemistich
* قافیه = the last word before the ردیف; its shared ending = حروف قافیه

For a ghazal/qasida the rhyming lines are both hemistichs of the first couplet plus the second
hemistich of every other couplet; in a masnavi each couplet rhymes internally.
Ganjoor's ``RhymeLetters`` gives a free evaluation target (scripts/eval_aruz.py --rhyme).
"""

from __future__ import annotations

from dataclasses import dataclass

from myllm.aruz.phonology import MARKS, normalize_verse


def _plain(word: str) -> str:
    """Letters only; «آ» = «ا» + ā, so آبی rhymes with تابی on «ابی»."""
    return "".join(ch for ch in word if ch not in MARKS and ch != "‌").replace("آ", "ا")


def _words(line: str) -> list[str]:
    return [_plain(w) for w in normalize_verse(line).split(" ") if _plain(w)]


def _common_suffix(strings: list[str]) -> str:
    if not strings:
        return ""
    s = min(strings, key=len)
    for k in range(len(s), 0, -1):
        tail = s[-k:]
        if all(x.endswith(tail) for x in strings):
            return tail
    return ""


@dataclass
class Rhyme:
    radif: str
    qafiye_words: list[str]
    letters: str

    @property
    def ok(self) -> bool:
        return bool(self.letters)


def find_rhyme(lines: list[str]) -> Rhyme:
    toks = [_words(line) for line in lines if line.strip()]
    if len(toks) < 2 or any(not t for t in toks):
        return Rhyme("", [], "")
    radif: list[str] = []
    k = 1
    while all(len(t) > k for t in toks) and len({t[-k] for t in toks}) == 1:
        radif.insert(0, toks[0][-k])
        k += 1
    qafiye = [t[-k] for t in toks]
    letters = _common_suffix(qafiye)
    if len(set(qafiye)) == 1:  # identical rhyme words: the repeated word is the refrain itself
        return Rhyme(" ".join([qafiye[0], *radif]), qafiye, "")
    return Rhyme(" ".join(radif), qafiye, letters)


def ghazal_rhyme_lines(verses: list[tuple[int, str, str]]) -> list[str]:
    """(couplet_index, position, text) -> rhyming hemistichs of a ghazal/qasida."""
    out = []
    for c, pos, text in verses:
        if (c == 0) or pos == "Left":
            out.append(text)
    return out



def format_rhyme(r: Rhyme) -> str:
    if not r.ok and not r.radif:
        return "قافیه‌ی مشترکی پیدا نشد."
    parts = []
    if r.ok:
        parts.append(f"قافیه: {'، '.join(dict.fromkeys(r.qafiye_words))} (حروف مشترک: «{r.letters}»)")
    if r.radif:
        parts.append(f"ردیف: «{r.radif}»")
    return " — ".join(parts)
