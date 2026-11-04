"""Evaluate metre identification on held-out Ganjoor poems (the number that decides the product).

    python scripts/eval_aruz.py                                    # no lexicon (rules only)
    python scripts/eval_aruz.py --lexicon data/processed/aruz_lexicon.json
    python scripts/eval_aruz.py --per-meter 80 --couplets          # couplet-level (both hemistichs)

* poems are split deterministically (myllm.aruz.lexicon.is_holdout) — the lexicon never saw these
* sampling is stratified by metre (Shahnameh alone would otherwise be half the test set)
* reports micro + macro accuracy with bootstrap CIs, per metre, per poet, the majority baseline,
  and accuracy by confidence bucket (is «اطمینان: بالا» trustworthy?)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from myllm.aruz.ganjoor import load_corpus
from myllm.aruz.lexicon import is_holdout, load_lexicon
from myllm.aruz.meters import meter_for_rhythm, same_meter
from myllm.aruz.scan import Scanner
from myllm.research.stats import bootstrap_ci


def eval_rhyme(root: Path) -> int:
    from myllm.aruz.rhyme import _plain, find_rhyme, ghazal_rhyme_lines

    n = ok = 0
    for f in sorted(root.rglob("*.json")):
        if "ghazal" not in str(f):
            continue
        poem = json.loads(f.read_text(encoding="utf-8"))
        gold = _plain(poem.get("RhymeLetters") or "").replace(" ", "")
        if not gold:
            continue
        verses = [(v["CoupletIndex"], v["Position"], v["Text"]) for v in poem["Verses"] if v.get("Position") in ("Right", "Left")]
        r = find_rhyme(ghazal_rhyme_lines(verses))
        n += 1
        ok += (r.letters + r.radif).replace(" ", "") == gold
    print(f"rhyme+radif exact match vs Ganjoor: {ok}/{n} = {ok / max(1, n):.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ganjoor")
    p.add_argument("--lexicon")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--per-meter", type=int, default=60)
    p.add_argument("--couplets", action="store_true", help="evaluate couplets (both hemistichs jointly)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/aruz/eval.json")
    p.add_argument("--rhyme", action="store_true", help="evaluate rhyme+radif on ghazals vs Ganjoor RhymeLetters")
    a = p.parse_args(argv)
    if a.rhyme:
        return eval_rhyme(Path(a.root))

    rows = [h for h in load_corpus(Path(a.root)) if is_holdout(h.poem_url, a.holdout)]
    rng = random.Random(a.seed)
    if a.couplets:
        pairs = defaultdict(dict)
        for h in rows:
            pairs[(h.poem_url, h.couplet)][h.position] = h
        units = [(v["Right"], v["Left"]) for v in pairs.values() if "Right" in v and "Left" in v]
    else:
        units = [(h,) for h in rows]
    by_meter = defaultdict(list)
    for u in units:
        m = meter_for_rhythm(u[0].rhythm)
        if m is not None:
            by_meter[m.name].append(u)
    sample = []
    for _name, us in sorted(by_meter.items()):
        rng.shuffle(us)
        sample.extend(us[: a.per_meter])
    scanner = Scanner(load_lexicon(a.lexicon) if a.lexicon else None)

    results, t0 = [], time.time()
    for u in sample:
        gold = meter_for_rhythm(u[0].rhythm)
        r = scanner.scan_couplet(u[0].text, u[1].text)[0] if a.couplets else scanner.scan(u[0].text)
        results.append({"gold": gold.name, "pred": r.meter.name if r.meter else None, "ok": same_meter(gold, r.meter),
                        "conf": r.confidence, "poet": u[0].poet, "text": " / ".join(x.text for x in u)})
    elapsed = time.time() - t0

    ok = [float(r["ok"]) for r in results]
    micro = bootstrap_ci(ok)
    per_meter = defaultdict(list)
    per_poet = defaultdict(list)
    for r in results:
        per_meter[r["gold"]].append(float(r["ok"]))
        per_poet[r["poet"]].append(float(r["ok"]))
    macro = sum(sum(v) / len(v) for v in per_meter.values()) / len(per_meter)
    majority = Counter(r["gold"] for r in results).most_common(1)[0][1] / len(results)
    buckets = {"high (>=0.7)": [], "mid (0.4-0.7)": [], "low (<0.4)": []}
    for r in results:
        key = "high (>=0.7)" if r["conf"] >= 0.7 else "mid (0.4-0.7)" if r["conf"] >= 0.4 else "low (<0.4)"
        buckets[key].append(float(r["ok"]))
    confusion = Counter((r["gold"], r["pred"]) for r in results if not r["ok"]).most_common(8)

    report = {
        "unit": "couplet" if a.couplets else "hemistich", "n": len(results), "lexicon": bool(a.lexicon),
        "micro_accuracy": [round(x, 4) for x in micro], "macro_accuracy_over_meters": round(macro, 4),
        "stratified_majority_baseline": round(majority, 4), "seconds_per_item": round(elapsed / max(1, len(results)), 3),
        "by_confidence": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4) if v else None} for k, v in buckets.items()},
        "per_meter": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(per_meter.items())},
        "per_poet": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(per_poet.items())},
        "top_confusions": [{"gold": g, "pred": pr, "count": c} for (g, pr), c in confusion],
        "errors_sample": [r for r in results if not r["ok"]][:25],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['unit']}s: n={report['n']}  micro acc={micro[0]:.3f} [{micro[1]:.3f}, {micro[2]:.3f}]  "
          f"macro={macro:.3f}  baseline={majority:.3f}  {report['seconds_per_item']}s/item")
    for k, v in report["by_confidence"].items():
        print(f"  confidence {k:<14} n={v['n']:<4} acc={v['acc']}")
    for k, v in report["per_meter"].items():
        print(f"  {v['acc']:.2f}  n={v['n']:<3} {k}")
    print("  top confusions:", [(c["gold"][:22], (c["pred"] or "None")[:22], c["count"]) for c in report["top_confusions"][:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
