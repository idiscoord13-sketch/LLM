"""Tune the ensemble weights (α = metre prior, β = neural classifier) on the DEV split only.

    python scripts/tune_aruz.py --workers 4 --classifier out/aruz/classifier.pt

1. compute the metre prior from TRAIN poems (one vote per poem)
2. sample DEV couplets twice: stratified by metre (every metre counts) and natural (real frequencies)
3. scan each unit ONCE, cache per-metre scanner costs + classifier log-probs
4. grid-search α, β offline; objective = mean(stratified acc, natural acc)
5. write data/processed/aruz_ensemble.json (used by myllm.aruz.engine.build_scanner)
Test poets are never touched here.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from myllm.aruz.engine import metre_prior
from myllm.aruz.ganjoor import load_corpus
from myllm.aruz.lexicon import is_holdout, load_lexicon
from myllm.aruz.meters import METERS, meter_for_rhythm, meter_key
from myllm.aruz.scan import Scanner

_W: dict = {}
_KEY = {m.pattern: meter_key(m) for m in METERS}


def _init(lexicon_path: str, classifier_path: str | None) -> None:
    _W["scanner"] = Scanner(load_lexicon(lexicon_path))
    _W["clf"] = None
    if classifier_path and Path(classifier_path).exists():
        import torch

        from myllm.aruz.classifier import MeterClassifier

        torch.set_num_threads(1)
        _W["clf"] = MeterClassifier.load(classifier_path)


def _features(texts: list[str]):
    sc = _W["scanner"]
    bests = [sc.score_meters(t)[0] for t in texts]
    pats = set().union(*[b.keys() for b in bests]) if bests else set()
    cost = {p: sum(b[p][0] if p in b else sc.MISSING_PENALTY for b in bests) for p in pats}
    nn = {}
    if _W["clf"] is not None:
        for lp in _W["clf"].logprobs(texts):
            for k, v in lp.items():
                nn[k] = nn.get(k, 0.0) + v
    return cost, nn


def _job(unit):
    gold, texts = unit
    return gold, *_features(texts)


def predict(cost: dict, nn: dict, prior: dict, alpha: float, beta: float, floor: float = -9.21) -> str | None:
    best, pick = float("inf"), None
    for p, c in cost.items():
        k = _KEY[p]
        total = c + beta * -nn.get(k, floor * 2) + alpha * -prior.get(k, floor)
        if total < best:
            best, pick = total, k
    return pick


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/ganjoor")
    ap.add_argument("--lexicon", default="data/processed/aruz_lexicon.json")
    ap.add_argument("--classifier", default="out/aruz/classifier.pt")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--per-meter", type=int, default=30)
    ap.add_argument("--natural", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/aruz_ensemble.json")
    a = ap.parse_args(argv)

    rows = load_corpus(Path(a.root))
    poem_keys, couplets = {}, defaultdict(dict)
    for h in rows:
        m = meter_for_rhythm(h.rhythm or "")
        if m is None:
            continue
        if not is_holdout(h.poem_url, a.holdout):
            poem_keys[h.poem_url] = meter_key(m)
        else:
            couplets[(h.poem_url, h.couplet)][h.position] = (h.text, meter_key(m))
    prior = metre_prior(list(poem_keys.values()))
    units = [(d["Right"][1], [d["Right"][0], d["Left"][0]]) for d in couplets.values() if "Right" in d and "Left" in d]
    rng = random.Random(a.seed)
    rng.shuffle(units)
    by = defaultdict(list)
    for u in units:
        by[u[0]].append(u)
    strat = [u for us in by.values() for u in us[: a.per_meter]]
    natural = units[: a.natural]
    print(f"prior from {len(poem_keys)} train poems; dev couplets: stratified {len(strat)}, natural {len(natural)}", flush=True)

    import multiprocessing as mp

    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.lexicon, a.classifier)) as pool:
        feats_s = pool.map(_job, strat, chunksize=8)
        feats_n = pool.map(_job, natural, chunksize=8)
    print(f"scanned in {time.time() - t0:.0f}s", flush=True)
    has_nn = any(f[2] for f in feats_s)

    def acc(feats, alpha, beta):
        return sum(predict(c, n, prior, alpha, beta) == g for g, c, n in feats) / max(1, len(feats))

    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0] if has_nn else [0.0]
    grid = []
    for al in alphas:
        for be in betas:
            s, n = acc(feats_s, al, be), acc(feats_n, al, be)
            grid.append(((s + n) / 2, s, n, al, be))
    grid.sort(reverse=True)
    for obj, s, n, al, be in grid[:8]:
        print(f"α={al:<5} β={be:<5} stratified={s:.4f} natural={n:.4f} mean={obj:.4f}")
    base = next(g for g in grid if g[3] == 0.0 and g[4] == 0.0)
    print(f"baseline (α=0, β=0): stratified={base[1]:.4f} natural={base[2]:.4f}")
    _, s, n, al, be = grid[0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"prior_weight": al, "classifier_weight": be, "prior": prior,
                                        "dev": {"stratified": s, "natural": n, "baseline": base[1:3]}},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
