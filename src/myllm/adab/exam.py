"""Printable exam builder for teachers: RTL HTML (print → PDF) with several versions and answer keys.

    python -m myllm.adab.exam --mix meter:6,ikhtiyar:2,taqti:2,history:10 --versions 2 \
        --title "آزمون علوم و فنون ادبی" --school "دبیرستان …" --minutes 40 --out out/exam/exam.html

Questions come from the local banks (موسیقی شعر: data/processed/aruz_exercises.jsonl, تاریخ ادبیات:
data/processed/history_bank.jsonl). Every version has the same questions in a different order with
shuffled options, so neighbours cannot copy; each version gets its own answer key and a worked-answer
page (پاسخ تشریحی). Open questions (تقطیع، قافیه) get answer lines and a full answer in the key.
"""

from __future__ import annotations

import argparse
import html
import random
from dataclasses import replace
from pathlib import Path

from myllm.aruz.exercises import Question, load_bank

LETTERS = ("الف", "ب", "ج", "د")
VERSION_NAMES = ("الف", "ب", "ج", "د")
GROUPS = {
    "history": {"work_author", "work_of", "not_by", "century", "style", "title", "chronology", "verse_poet"},
    "aruz": {"meter", "ikhtiyar", "taqti", "rhyme"},
}
_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x) -> str:
    return str(x).translate(_FA)


def parse_mix(spec: str) -> list[tuple[set[str], int]]:
    """'meter:6,history:10' -> [({'meter'}, 6), ({...history kinds}, 10)]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, n = part.partition(":")
        out.append((GROUPS.get(name, {name}), int(n or 1)))
    return out


def select(bank: list[Question], mix: list[tuple[set[str], int]], rng: random.Random) -> list[Question]:
    chosen, used = [], set()
    for kinds, n in mix:
        pool = [q for q in bank if q.kind in kinds and id(q) not in used]
        rng.shuffle(pool)
        verses = {tuple(q.verse) for q in chosen if q.verse}
        picks = []
        for q in pool:  # one question per verse, so a couplet does not give away another answer
            if q.verse and tuple(q.verse) in verses:
                continue
            picks.append(q)
            verses.add(tuple(q.verse))
            if len(picks) == n:
                break
        if len(picks) < n:
            raise ValueError(f"bank has only {len(picks)} usable questions of kinds {sorted(kinds)}; asked for {n}")
        chosen += picks
        used |= {id(q) for q in picks}
    return chosen


def shuffle_version(questions: list[Question], seed: int) -> list[Question]:
    """Same questions, new order, options shuffled (answer index remapped)."""
    rng = random.Random(seed)
    qs = questions[:]
    rng.shuffle(qs)
    out = []
    for q in qs:
        if q.answer is None:
            out.append(q)
            continue
        order = list(range(len(q.options)))
        rng.shuffle(order)
        options = [q.options[i] for i in order]
        out.append(replace(q, options=options, answer=order.index(q.answer)))
    return out


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
:root { --ink: #1d1d1f; --muted: #5f6368; --line: #c9ccd1; --paper: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.9 "Vazirmatn", Tahoma, sans-serif; }
.page { max-width: 190mm; margin: 0 auto; padding: 12mm 10mm; }
header { border: 1.5px solid var(--ink); border-radius: 6px; padding: 8px 14px; margin-bottom: 14px; }
header h1 { font-size: 19px; margin: 0 0 4px; text-align: center; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 22px; color: var(--muted); font-size: 13px; justify-content: center; }
.student { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px 16px; margin-top: 8px; font-size: 13px; }
.student span { border-bottom: 1px dotted var(--line); padding-bottom: 2px; }
.q { break-inside: avoid; page-break-inside: avoid; border-bottom: 1px solid var(--line); padding: 10px 0; }
.q .n { font-weight: 700; margin-left: 6px; }
.verse { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 28px; margin: 6px 6% 4px; text-align: center; }
.verse.one { grid-template-columns: 1fr; }
.opts { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 24px; margin-top: 4px; }
.lines div { border-bottom: 1px dotted var(--line); height: 26px; }
.key { page-break-before: always; break-before: page; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
td, th { border: 1px solid var(--line); padding: 3px 8px; text-align: center; }
.worked p { margin: 4px 0; }
@media print { .page { padding: 0; } @page { size: A4; margin: 12mm; } }
@media (max-width: 600px) { .verse, .opts, .student { grid-template-columns: 1fr; } }
"""


def _verse_html(verse: list[str]) -> str:
    if not verse:
        return ""
    cls = "verse" if len(verse) == 2 else "verse one"
    return f'<div class="{cls}">' + "".join(f"<div>{html.escape(v)}</div>" for v in verse) + "</div>"


def render_html(questions: list[Question], title: str, school: str = "", minutes: int | None = None,
                version: str = "") -> str:
    e = html.escape
    meta = [f"نسخه‌ی {e(version)}" if version else "", e(school),
            f"زمان: {fa(minutes)} دقیقه" if minutes else "", f"تعداد سؤال: {fa(len(questions))}"]
    parts = [f'<section class="page"><header><h1>{e(title)}</h1>',
             '<div class="meta">' + "".join(f"<span>{m}</span>" for m in meta if m) + "</div>",
             '<div class="student"><span>نام و نام خانوادگی:</span><span>کلاس:</span><span>تاریخ:</span></div></header>']
    for k, q in enumerate(questions, 1):
        body = [f'<div class="q"><span class="n">{fa(k)}.</span>{e(q.prompt)}', _verse_html(q.verse)]
        if q.answer is not None:
            body.append('<div class="opts">' + "".join(f"<div>{LETTERS[i]}) {e(o)}</div>" for i, o in enumerate(q.options))
                        + "</div>")
        else:
            body.append('<div class="lines"><div></div><div></div></div>')
        parts.append("".join(body) + "</div>")
    parts.append("</section>")
    rows = "".join(f"<tr><td>{fa(k)}</td><td>{LETTERS[q.answer] if q.answer is not None else 'تشریحی'}</td></tr>"
                   for k, q in enumerate(questions, 1))
    worked = "".join(
        f"<p><b>{fa(k)}.</b> {e(q.answer_text if q.answer is None else LETTERS[q.answer] + ') ' + q.options[q.answer])}"
        + (f" — {e(q.explanation)}" if q.explanation else "") + "</p>"
        for k, q in enumerate(questions, 1))
    parts.append(f'<section class="page key"><h2>پاسخ‌نامه{(" — نسخه‌ی " + e(version)) if version else ""}</h2>'
                 f"<table><tr><th>سؤال</th><th>پاسخ</th></tr>{rows}</table>"
                 f'<h2>پاسخ تشریحی</h2><div class="worked">{worked}</div></section>')
    return "".join(parts)


def build_exam(bank: list[Question], mix: str, versions: int = 1, seed: int = 42, title: str = "آزمون",
               school: str = "", minutes: int | None = None) -> tuple[str, list[list[Question]]]:
    rng = random.Random(seed)
    base = select(bank, parse_mix(mix), rng)
    sets = [shuffle_version(base, seed + v) for v in range(versions)]
    body = "".join(render_html(qs, title, school, minutes, VERSION_NAMES[v] if versions > 1 else "")
                   for v, qs in enumerate(sets))
    doc = (f'<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
           f"<style>{_CSS}</style></head><body>{body}</body></html>")
    return doc, sets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mix", default="meter:6,ikhtiyar:2,taqti:2,history:10")
    p.add_argument("--versions", type=int, default=2, choices=range(1, 5))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", default="آزمون علوم و فنون ادبی")
    p.add_argument("--school", default="")
    p.add_argument("--minutes", type=int, default=40)
    p.add_argument("--banks", default="data/processed/aruz_exercises.jsonl,data/processed/history_bank.jsonl")
    p.add_argument("--out", default="out/exam/exam.html")
    a = p.parse_args(argv)
    bank = [q for path in a.banks.split(",") for q in load_bank(path)]
    doc, sets = build_exam(bank, a.mix, a.versions, a.seed, a.title, a.school, a.minutes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"{len(sets[0])} questions × {len(sets)} versions -> {a.out} (open in a browser and print to PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
