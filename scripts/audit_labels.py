"""List likely *label* errors in Ganjoor: lines where the engine is confident and disagrees with the label.

    python scripts/audit_labels.py --root data/raw/ganjoor_newpoets --holdout 1.0 --workers 4 \
        --out out/aruz/label_audit.md

A "near 100%" claim needs this: if Ganjoor's metre label is wrong, a correct engine is counted as
wrong. This script does NOT change any number — it produces a review list. A human (ideally a
literature teacher) marks each row as «برچسب غلط است» or «موتور غلط است»; only then may the
evaluation exclude confirmed label errors (report both numbers).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from myllm.aruz.ganjoor import load_corpus
from myllm.aruz.lexicon import is_holdout
from myllm.aruz.meters import meter_for_rhythm, same_meter

_W: dict = {}


def _init(lexicon, ensemble, classifier) -> None:
    from myllm.aruz.engine import build_scanner

    _W["sc"] = build_scanner(lexicon, ensemble, classifier, threads=1)


def _job(unit):
    url, texts, rhythm = unit
    rs = _W["sc"].scan_poem(texts)
    gold = meter_for_rhythm(rhythm)
    r = rs[0]
    return url, texts, gold.name if gold else rhythm, r.meter.name if r.meter else None, r.confidence, \
        same_meter(gold, r.meter), " | ".join(r.syllables)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ganjoor")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--couplets", type=int, default=3, help="couplets per poem to scan jointly")
    p.add_argument("--max-poems", type=int, default=600)
    p.add_argument("--min-conf", type=float, default=0.9)
    p.add_argument("--lexicon", default="data/processed/aruz_lexicon.json")
    p.add_argument("--ensemble", default="data/processed/aruz_ensemble.json")
    p.add_argument("--classifier", default="out/aruz/classifier.pt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="out/aruz/label_audit.md")
    a = p.parse_args(argv)
    poems = defaultdict(list)
    rhythm = {}
    for h in load_corpus(Path(a.root)):
        if is_holdout(h.poem_url, a.holdout) and meter_for_rhythm(h.rhythm or "") is not None:
            poems[h.poem_url].append(h.text)
            rhythm[h.poem_url] = h.rhythm
    units = [(u, t[: 2 * a.couplets], rhythm[u]) for u, t in poems.items() if len(t) >= 2 * a.couplets]
    random.Random(0).shuffle(units)
    units = units[: a.max_poems]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.lexicon, a.ensemble, a.classifier)) as pool:
        rows = pool.map(_job, units, chunksize=4)
    flagged = [r for r in rows if not r[5] and r[4] >= a.min_conf]
    wrong = [r for r in rows if not r[5]]
    lines = ["# ممیزی برچسب‌های وزن گنجور", "",
             f"{len(rows)} شعر بررسی شد؛ {len(wrong)} اختلاف با برچسب؛ {len(flagged)} مورد با اطمینان ≥ {a.min_conf} "
             "(کاندیدای «برچسب غلط»).", "",
             "| # | شعر | برچسب گنجور | تشخیص موتور | اطمینان | مصراع اول | تقطیع | داوری (برچسب غلط / موتور غلط) |",
             "|---|---|---|---|---|---|---|---|"]
    for k, (url, texts, gold, pred, conf, _ok, syl) in enumerate(flagged, 1):
        lines.append(f"| {k} | {url} | {gold} | {pred} | {conf:.2f} | {texts[0]} | {syl} | |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} poems, {len(wrong)} disagreements, {len(flagged)} confident -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
