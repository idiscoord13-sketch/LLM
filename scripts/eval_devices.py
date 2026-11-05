"""Evaluate rule-based devices against Ganjoor's LiteraryTags (MIT-licensed academic corpus, Hafez).

    python scripts/eval_devices.py --root data/raw/ganjoor/hafez

Gold: tags with Origin=AcademicCorpus (persian-literary-corpus-1.0.0, MIT; Raji, Alikhani, de Melo,
Stone 2023). Only devices this engine claims to detect by rule are scored:
  * جناس  — a predicted pair is correct if both words are the gold pair's words (same couplet)
  * تشبیه — a predicted simile (ادات) is counted correct if the couplet has a gold تشبیه tag
The corpus is not exhaustive (annotators did not tag every device), so precision against it is a
LOWER bound; the review sheet (--sheet) lets a teacher check the "false" positives by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from myllm.adab.devices import find_jenas, find_simile, tokenize
from myllm.aruz.rhyme import find_rhyme

JENAS, TASHBIH = 6, 1


def gold_tags(poem: dict) -> dict[tuple[int, int], list[set[tuple[int, int]]]]:
    """(device, couplet) -> list of word-position sets (misra, word) per tag."""
    out = defaultdict(list)
    for t in poem.get("LiteraryTags") or []:
        if t.get("Origin") != "AcademicCorpus":
            continue
        pos = {(p["Misra"], w) for p in t.get("Parts", []) if p.get("Misra") in (1, 2) and p.get("WordStart")
               for w in range(p["WordStart"], p.get("WordEnd", p["WordStart"]) + 1)}
        out[(t["DeviceId"], t["CoupletIndex"])].append(pos)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/ganjoor/hafez")
    ap.add_argument("--sheet", default="out/adab/devices_review.md")
    ap.add_argument("--top1", action="store_true", help="only the most salient جناس per couplet")
    a = ap.parse_args(argv)
    tp = fp = fn = 0
    s_tp = s_fp = 0
    gold_simile_couplets = 0
    review = []
    for f in sorted(Path(a.root).rglob("*.json")):
        poem = json.loads(f.read_text(encoding="utf-8"))
        gold = gold_tags(poem)
        if not gold:
            continue
        couplets = defaultdict(dict)
        for v in poem.get("Verses", []):
            if v.get("Position") in ("Right", "Left"):
                couplets[v["CoupletIndex"]][v["Position"]] = v["Text"]
        for c, d in couplets.items():
            if "Right" not in d or "Left" not in d:
                continue
            words = tokenize([d["Right"], d["Left"]])
            radif = set(find_rhyme([d["Right"], d["Left"]]).radif.split())
            pred = [p for p in find_jenas(words, exclude=radif) if not p.kind.startswith("جناس تام")]
            if a.top1:
                pred = pred[:1]
            golds = gold.get((JENAS, c), [])
            matched = set()
            for p in pred:
                hit = next((k for k, g in enumerate(golds) if p.positions() <= g or g <= p.positions()), None)
                if hit is not None and hit not in matched:
                    tp += 1
                    matched.add(hit)
                else:
                    fp += 1
                    review.append((poem["FullUrl"], c, d["Right"], d["Left"], p.kind, " / ".join(w.plain for w in p.words)))
            fn += len(golds) - len(matched)
            sims = find_simile(words)
            has_gold = bool(gold.get((TASHBIH, c)))
            gold_simile_couplets += has_gold
            for _s in sims:
                if has_gold:
                    s_tp += 1
                else:
                    s_fp += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"جناس (اختلافی/افزایشی/حرکتی): precision>={prec:.3f}  recall={rec:.3f}  (tp={tp} fp={fp} fn={fn})")
    print(f"تشبیه با ادات: {s_tp}/{s_tp + s_fp} predictions fall in couplets with a gold تشبیه "
          f"(lower bound on precision; gold تشبیه couplets: {gold_simile_couplets})")
    Path(a.sheet).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# بازبینی «جناس»های بی‌برچسب (برای داوری دبیر)", "",
             "| شعر | بیت | مصراع ۱ | مصراع ۲ | نوع | واژه‌ها | داوری |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {u} | {c} | {r} | {left} | {k} | {w} | |" for u, c, r, left, k, w in review]
    Path(a.sheet).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"review sheet with {len(review)} unmatched predictions -> {a.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
