"""Fetch a large, poet-diverse Ganjoor corpus for training, plus untouched test poets.

    python scripts/fetch_ganjoor.py --per-poet 150                 # train poets -> data/raw/ganjoor
    python scripts/fetch_ganjoor.py --test2 --per-poet 80          # test2 poets -> data/raw/ganjoor_test2

Evaluation discipline (docs/product/aruz_big_update.md):
* TEST1 poets (data/raw/ganjoor_newpoets) and TEST2 poets are never downloaded into the training root;
* TEST2 is evaluated exactly once, for the final report.
Files are slimmed (no summaries/HTML) and git-ignored; Ganjoor's data has no LICENSE — keep it local.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from myllm.aruz.ganjoor import _fetch_json, download

TEST1 = ["attar", "sanaee", "jami", "eraghi", "khajoo", "vahshi", "shahriar", "saeb", "rahi", "forooghi", "hatef", "anvari"]
TEST2 = ["masood", "manoochehri", "farrokhi", "naserkhosro", "ouhadi", "seyf", "salman", "khojandi", "iqbal", "shahnematollah"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-poet", type=int, default=150)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--test2", action="store_true", help="download only the TEST2 poets into --out (default data/raw/ganjoor_test2)")
    p.add_argument("--out")
    a = p.parse_args(argv)
    manifest = _fetch_json("manifest.json")
    slugs = [poet["FullUrl"].strip("/") for poet in manifest["Poets"]]
    if a.test2:
        chosen, out = [s for s in slugs if s in TEST2], Path(a.out or "data/raw/ganjoor_test2")
    else:
        chosen, out = [s for s in slugs if s not in TEST1 and s not in TEST2], Path(a.out or "data/raw/ganjoor")
    print(f"{len(chosen)} poets -> {out}", flush=True)
    t0 = time.time()
    total = 0
    for k, slug in enumerate(chosen, 1):
        try:
            n = download(f"/{slug}", out, max_poems=a.per_poet, threads=a.threads)
        except Exception as exc:  # a poet with an odd category tree must not stop the crawl
            print(f"  skip {slug}: {exc}", flush=True)
            continue
        total += n
        print(f"[{k}/{len(chosen)}] {slug}: {n} poems (total {total}, {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
