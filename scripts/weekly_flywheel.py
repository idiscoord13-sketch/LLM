"""One turn of the weekly data flywheel (roadmap §7.3, all 8 steps).

    python scripts/weekly_flywheel.py --config configs/flywheel.yaml --dry-run
    python scripts/weekly_flywheel.py --config configs/flywheel.yaml --import data/samples/flywheel_logs.jsonl --dry-run
    python scripts/weekly_flywheel.py --config configs/flywheel.yaml          # trains (needs a GPU)

1 log (consent)  2 anonymise  3 signals  4 filter/dedup  5 build SFT/DPO/KTO (+replay)
6 periodic training  7 eval + regression gate  8 register -> canary (rollback via release.py)

With ``--dry-run`` (or no GPU) training is skipped and the gate runs on existing
results if provided, so the whole loop can be rehearsed on a laptop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from myllm.config import get, load_config
from myllm.flywheel.build_dataset import build, write_all
from myllm.flywheel.filters import apply_filters
from myllm.flywheel.gate import check
from myllm.flywheel.release import Registry
from myllm.flywheel.store import FlywheelStore
from myllm.train.modeling import cuda_available
from myllm.utils import read_jsonl


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/flywheel.yaml")
    p.add_argument("--import", dest="import_file", help="load pre-collected logs through the consent/PII gate first")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--new-results", help="eval results of the candidate (skip training/eval)")
    p.add_argument("--baseline-results", help="eval results of the stable model")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    week = time.strftime("%Y-W%V")
    out_dir = Path(get(cfg, "output_dir", "data/flywheel/weeks")) / week

    store = FlywheelStore(get(cfg, "store_dir", "data/flywheel"))
    if a.import_file:
        print(f"[1-2] imported {store.import_rows(read_jsonl(a.import_file))} consented conversations (PII scrubbed)")
    rows = store.load()
    print(f"[1-2] {len(rows)} consented conversations in store")

    kept, fstats = apply_filters(rows, get(cfg, "filters", {}))
    print(f"[3-4] filters: {fstats}")
    datasets = build(kept, get(cfg, "build", {}))
    counts = write_all(datasets, out_dir, get(cfg, "replay", {}))
    print(f"[5]   datasets -> {out_dir}: {counts}")

    min_rows = int(get(cfg, "min_rows_to_train", 200))
    can_train = cuda_available() and not a.dry_run and counts.get("kto", 0) >= min_rows
    new_results = a.new_results
    if can_train:
        for step in get(cfg, "train_steps", []):
            run([sys.executable, "-m", f"myllm.train.{step['trainer']}", "--config", step["config"],
                 "--set", f"data.train_files=[{out_dir / (step['data'] + '.jsonl')}]",
                 "--set", f"output_dir={out_dir / step['trainer']}", *step.get("overrides", [])])
        run([sys.executable, "-m", "myllm.eval.run_eval", "--config", get(cfg, "eval_config"),
             "--set", f"output_dir={out_dir / 'eval'}"])
        new_results = str(out_dir / "eval" / "results.json")
    else:
        why = "dry-run" if a.dry_run else ("no GPU" if not cuda_available() else f"< {min_rows} rows")
        print(f"[6]   training skipped ({why})")

    if not new_results:
        print("[7]   no candidate results -> nothing to gate; done")
        return 0
    new = json.loads(Path(new_results).read_text(encoding="utf-8"))
    base = json.loads(Path(a.baseline_results).read_text(encoding="utf-8")) if a.baseline_results else None
    ok, lines = check(new, base, get(cfg, "gate", {}))
    print("[7]   regression gate:\n      " + "\n      ".join(lines))
    if not ok:
        print("GATE FAIL -> candidate is NOT released")
        return 1
    if a.dry_run:
        print("[8]   GATE PASS (dry-run: not registering)")
        return 0
    reg = Registry(get(cfg, "registry", "data/flywheel/registry.json"))
    version = f"{week}-{int(time.time())}"
    reg.register(version, get(cfg, "release.upstream"), get(cfg, "release.model", version), new.get("metrics"))
    reg.start_canary(version, float(get(cfg, "release.canary_percent", 10)))
    print(f"[8]   registered {version} as canary at {get(cfg, 'release.canary_percent', 10)}% "
          "— watch live metrics, then `release promote` or `release rollback`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
