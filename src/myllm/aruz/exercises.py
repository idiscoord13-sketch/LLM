"""Exercise generator: unlimited practice questions with *verified* answers.

    python -m myllm.aruz.exercises --root data/raw/ganjoor --n 40 --out data/processed/aruz_exercises.jsonl \
        --md out/aruz/worksheet.md

A couplet becomes a question only when (a) the engine reads it with high confidence and
(b) its metre agrees with Ganjoor's label — two independent sources, so the answer key is
trustworthy. Question types:
  * meter     — «وزن این بیت کدام است؟» four options; distractors are the metres the engine
                itself found closest (the realistic confusions, not random ones)
  * taqti     — «مصراع اول را تقطیع هجایی کنید» (answer: syllables + – / U)
  * ikhtiyar  — «کدام اختیار شاعری در این بیت به کار رفته؟» (only when one was used)
  * rhyme     — «قافیه و ردیف مطلع را مشخص کنید» (ghazal openings)
Questions are derived from Ganjoor's texts: keep the bank local until Ganjoor's license is clear.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from myllm.aruz.meters import METERS, meter_for_rhythm, same_meter
from myllm.aruz.rhyme import find_rhyme
from myllm.aruz.scan import FA_NOTES, Scanner

LETTERS = "الف ب ج د".split()


@dataclass
class Question:
    kind: str
    prompt: str
    verse: list[str]
    options: list[str] = field(default_factory=list)
    answer: int | None = None  # index into options (multiple choice)
    answer_text: str = ""
    explanation: str = ""
    source: str = ""


def _signs(weights: list[str]) -> str:
    return " ".join("–" if w == "-" else ("– U" if w == "-u" else "U") for w in weights)


def _meter_options(r, rng: random.Random) -> tuple[list[str], int] | None:
    correct = f"{r.meter.feet_label()} ({r.meter.name})"
    names = [n for n, _ in r.ranking[1:]]
    pool = [m for m in METERS if m.name in names and m.family != r.meter.family]
    pool += [m for m in METERS if m not in pool and m.family != r.meter.family]
    seen, distractors = {correct}, []
    for m in pool:
        label = f"{m.feet_label()} ({m.name})"
        if label not in seen:
            seen.add(label)
            distractors.append(label)
        if len(distractors) == 3:
            break
    opts = [correct, *distractors]
    rng.shuffle(opts)
    return opts, opts.index(correct)


def questions_for(scanner: Scanner, first: str, second: str, gold_rhythm: str | None, rng: random.Random,
                  source: str = "", opening: bool = False, min_conf: float = 0.9) -> list[Question]:
    r1, r2 = scanner.scan_couplet(first, second)
    if not r1.ok or r1.confidence < min_conf:
        return []
    if gold_rhythm and not same_meter(meter_for_rhythm(gold_rhythm), r1.meter):
        return []  # engine and Ganjoor disagree -> not trustworthy enough for an answer key
    verse = [first, second]
    out = []
    opts, ans = _meter_options(r1, rng)
    out.append(Question("meter", "وزن این بیت کدام است؟", verse, opts, ans, opts[ans],
                        f"تقطیع مصراع اول: {' | '.join(r1.syllables)}", source))
    out.append(Question("taqti", "مصراع اول را تقطیع هجایی کنید و نشانه‌ی هجاها را بنویسید.", verse,
                        answer_text=f"{' | '.join(r1.syllables)}\n{_signs(r1.weights)}",
                        explanation=f"وزن: {r1.meter.feet_label()}", source=source))
    used = list(dict.fromkeys(n.split(" («")[0] for n in r1.ikhtiyarat + r2.ikhtiyarat))
    if used:
        correct = used[0]
        others = [v for v in FA_NOTES.values() if v not in used]
        rng.shuffle(others)
        opts = [correct, *others[:3]]
        rng.shuffle(opts)
        detail = "، ".join(dict.fromkeys(r1.ikhtiyarat + r2.ikhtiyarat))
        out.append(Question("ikhtiyar", "کدام اختیار شاعری در این بیت به کار رفته است؟", verse, opts,
                            opts.index(correct), correct, detail, source))
    if opening:
        rh = find_rhyme(verse)
        if rh.ok:
            ans_text = f"قافیه: {'، '.join(dict.fromkeys(rh.qafiye_words))} (حروف: «{rh.letters}»)"
            if rh.radif:
                ans_text += f" — ردیف: «{rh.radif}»"
            out.append(Question("rhyme", "قافیه و ردیف این بیت (مطلع) را مشخص کنید.", verse,
                                answer_text=ans_text, source=source))
    return out


def generate(scanner: Scanner, root: Path, n: int, seed: int = 42, kinds: set[str] | None = None,
             holdout: float | None = None) -> list[Question]:
    from myllm.aruz.ganjoor import iter_poem_files

    rng = random.Random(seed)
    files = list(iter_poem_files(root))
    rng.shuffle(files)
    bank: list[Question] = []
    per_meter: dict[str, int] = defaultdict(int)
    for f in files:
        poem = json.loads(f.read_text(encoding="utf-8"))
        rhythm = (poem.get("Metre") or {}).get("Rhythm")
        verses = [v for v in poem.get("Verses", []) if v.get("Position") in ("Right", "Left")]
        couplets = defaultdict(dict)
        for v in verses:
            couplets[v["CoupletIndex"]][v["Position"]] = v["Text"]
        idx = [c for c, d in couplets.items() if "Right" in d and "Left" in d]
        if not idx:
            continue
        c = 0 if 0 in idx else rng.choice(idx)
        m = meter_for_rhythm(rhythm or "")
        if m is not None and per_meter[m.name] >= max(2, n // 8):
            continue  # keep the bank diverse across metres
        qs = questions_for(scanner, couplets[c]["Right"], couplets[c]["Left"], rhythm, rng,
                           source=poem.get("FullUrl", ""), opening=c == 0 and "ghazal" in poem.get("FullUrl", ""))
        qs = [q for q in qs if kinds is None or q.kind in kinds]
        if qs:
            per_meter[m.name if m else "?"] += 1
            bank.extend(qs)
        if len(bank) >= n:
            break
    return bank[:n]


def to_markdown(bank: list[Question]) -> str:
    lines = ["# برگه‌ی تمرین موسیقی شعر", ""]
    for k, q in enumerate(bank, 1):
        lines.append(f"**{k}.** {q.prompt}")
        lines.append("")
        lines.append("> " + "  \n> ".join(q.verse))
        lines.append("")
        for i, o in enumerate(q.options):
            lines.append(f"{LETTERS[i]}) {o}  ")
        lines.append("")
    lines += ["---", "## پاسخ‌نامه", ""]
    for k, q in enumerate(bank, 1):
        ans = f"{LETTERS[q.answer]}) {q.options[q.answer]}" if q.answer is not None else q.answer_text.replace("\n", " — ")
        lines.append(f"{k}. {ans}" + (f" — {q.explanation}" if q.explanation else ""))
    return "\n".join(lines) + "\n"


def load_bank(path: str | Path) -> list[Question]:
    if not Path(path).exists():
        return []
    return [Question(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    from myllm.aruz.lexicon import load_lexicon

    p = argparse.ArgumentParser(description="generate verified prosody exercises")
    p.add_argument("--root", default="data/raw/ganjoor")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--kinds", default="meter,taqti,ikhtiyar,rhyme")
    p.add_argument("--lexicon", default="data/processed/aruz_lexicon.json")
    p.add_argument("--out", default="data/processed/aruz_exercises.jsonl")
    p.add_argument("--md", default="out/aruz/worksheet.md")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    scanner = Scanner(load_lexicon(a.lexicon))
    bank = generate(scanner, Path(a.root), a.n, a.seed, set(a.kinds.split(",")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(asdict(q), ensure_ascii=False) for q in bank) + "\n", encoding="utf-8")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(to_markdown(bank), encoding="utf-8")
    print(f"{len(bank)} questions -> {a.out} and {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
