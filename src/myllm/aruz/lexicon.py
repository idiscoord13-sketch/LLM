"""Learn a word-weight lexicon from metre-labelled verse (weak supervision).

Idea: Ganjoor tells us the metre of every poem but not how each word is pronounced.
Scanning a hemistich *restricted to its known metre* usually leaves one best reading, which
reveals the syllable shape of every word in it (e.g. «کند» = ko-nad, «بشنو» = beš-no).
Counting those choices over tens of thousands of hemistichs yields a lexicon of short-vowel
patterns — the knowledge a human reader brings — without anyone typing a dictionary.
A second pass rescans with the lexicon (EM-style) and refines it.

    python -m myllm.aruz.lexicon learn --root data/raw/ganjoor --out data/processed/aruz_lexicon.json \
        --max-hemistichs 40000 --iterations 2 --holdout 0.2

The lexicon is derived from Ganjoor's files — keep it local until their license is clear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from myllm.aruz.ganjoor import Hemistich, load_corpus
from myllm.aruz.meters import meter_for_rhythm
from myllm.aruz.phonology import split_words


def is_holdout(poem_url: str, fraction: float) -> bool:
    """Deterministic poem-level split so train/test never share a poem."""
    h = int(hashlib.sha1(poem_url.encode("utf-8")).hexdigest(), 16) % 1000
    return h < fraction * 1000


def word_signatures(scanner, hem: Hemistich, max_cost: float) -> list[tuple[str, str]] | None:
    """(part, signature) pairs from the best scan restricted to the labelled metre."""
    meter = meter_for_rhythm(hem.rhythm or "")
    if meter is None:
        return None
    restrict = [m for m in scanner.meters if m.family == "رباعی"] if meter.family == "رباعی" else [meter]
    saved = scanner.meters
    scanner.meters = restrict
    try:
        best, _ = scanner.score_meters(hem.text)
    finally:
        scanner.meters = saved
    if not best:
        return None
    cost, hyp, _ops = min(best.values(), key=lambda v: v[0])
    if cost > max_cost:
        return None
    # recover which readings were chosen: re-derive per-part signatures from the syllable spans
    out = []
    parts = [p for w in split_words(hem.text) for p in w]
    sigs = _segment(scanner, parts, hyp)
    if sigs is None:
        return None
    for part, sig in zip(parts, sigs):
        if part != "و":
            out.append((part, sig))
    return out


def _segment(scanner, parts: list[str], hyp) -> list[str] | None:
    """Greedy match of each part's readings against the chosen syllable sequence (shapes only)."""
    shapes = [("A" if s.long else "V") + str(len(s.coda)) for s in hyp.syls]
    i, sigs = 0, []
    for part in parts:
        if part == "و":
            sigs.append("و")
            i += 1
            continue
        found = None
        for r in scanner.readings(part):
            sig = r.signature.split(".")
            n = len(sig)
            window = shapes[i : i + n]
            if len(window) == n and all(_compatible(a, b) for a, b in zip(sig, window)):
                found = (r.signature, n)
                break
            # izafe adds one syllable; linking moves one coda consonant
            window = shapes[i : i + n + 1]
            if len(window) == n + 1 and all(_compatible(a, b) for a, b in zip(sig, window[:n])):
                found = (r.signature, n + 1)
                break
        if found is None:
            return None
        sigs.append(found[0])
        i += found[1]
    return sigs if i == len(shapes) else None


def _compatible(lexical: str, surface: str) -> bool:
    """Lexical shape vs surface shape after linking (coda may lose one consonant) / lengthening."""
    lv, lc = lexical[0], int(lexical[1])
    sv, sc = surface[0], int(surface[1])
    return (lv == sv or (lv == "V" and sv == "A")) and (sc == lc or sc == lc - 1)


_WORKER: dict = {}


def _init_worker(lexicon: dict, max_cost: float) -> None:
    from myllm.aruz.scan import Scanner

    _WORKER["scanner"] = Scanner(lexicon)
    _WORKER["max_cost"] = max_cost


def _signatures_job(h: Hemistich):
    return word_signatures(_WORKER["scanner"], h, _WORKER["max_cost"])


def learn(hems: list[Hemistich], iterations: int = 2, max_cost: float = 2.0, min_count: int = 2,
          log_every: int = 5000, workers: int = 1) -> dict[str, dict[str, float]]:
    """EM-style lexicon learning; ``workers > 1`` scans in parallel processes (same result, faster)."""
    lexicon: dict[str, dict[str, float]] = {}
    for it in range(iterations):
        counts: dict[str, Counter] = defaultdict(Counter)
        used, t0 = 0, time.time()
        if workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(lexicon, max_cost)) as pool:
                results = pool.imap(_signatures_job, hems, chunksize=64)
                for k, pairs in enumerate(results):
                    used += _count(pairs, counts)
                    _log(log_every, it, k, len(hems), used, t0)
        else:
            _init_worker(lexicon, max_cost)
            for k, h in enumerate(hems):
                used += _count(_signatures_job(h), counts)
                _log(log_every, it, k, len(hems), used, t0)
        lexicon = {}
        for part, c in counts.items():
            total = sum(c.values())
            if total < min_count:
                continue
            lexicon[part] = {sig: round(n / total, 4) for sig, n in c.most_common(4) if n / total >= 0.05}
        print(f"iteration {it + 1}: {used}/{len(hems)} hemistichs usable, lexicon size {len(lexicon)}", flush=True)
    return lexicon


def _count(pairs, counts) -> int:
    if not pairs:
        return 0
    for part, sig in pairs:
        counts[part][sig] += 1
    return 1


def _log(every: int, it: int, k: int, n: int, used: int, t0: float) -> None:
    if every and (k + 1) % every == 0:
        print(f"  iter {it + 1}: {k + 1}/{n} scanned, {used} used, {time.time() - t0:.0f}s", flush=True)


def save_lexicon(lex: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")


def load_lexicon(path: str | Path | None) -> dict[str, dict[str, float]]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="learn a word-weight lexicon from Ganjoor")
    sub = p.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("learn")
    lp.add_argument("--root", default="data/raw/ganjoor")
    lp.add_argument("--out", default="data/processed/aruz_lexicon.json")
    lp.add_argument("--max-hemistichs", type=int, default=40000)
    lp.add_argument("--iterations", type=int, default=2)
    lp.add_argument("--holdout", type=float, default=0.2, help="fraction of poems kept out for evaluation")
    lp.add_argument("--per-meter-cap", type=int, default=4000, help="balance metres (Shahnameh dominates)")
    lp.add_argument("--workers", type=int, default=1, help="parallel scanning processes")
    lp.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    rows = [h for h in load_corpus(Path(a.root)) if not is_holdout(h.poem_url, a.holdout)]
    import random

    random.Random(a.seed).shuffle(rows)  # sample across poets/poems, not the first files on disk
    per: Counter = Counter()
    picked = []
    for h in rows:
        if per[h.rhythm] < a.per_meter_cap:
            per[h.rhythm] += 1
            picked.append(h)
    picked = picked[: a.max_hemistichs]
    print(f"learning from {len(picked)} hemistichs ({len(per)} metres, holdout {a.holdout:.0%} of poems excluded)")
    lex = learn(picked, iterations=a.iterations, workers=a.workers)
    save_lexicon(lex, a.out)
    print(f"saved {len(lex)} entries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
