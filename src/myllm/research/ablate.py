"""Ablation runner: variants x seeds -> registry -> Markdown report with uncertainty.

    python -m myllm.research.ablate --config configs/research/idea4_mixing.yaml
    python -m myllm.research.ablate --config configs/smoke/ablation.yaml

Config::

    name: idea4_mixing
    command: [python, -m, myllm.tinygpt.pretrain, --config, configs/tiny_gpt.yaml]
    metric: {file: "{output_dir}/log.jsonl", key: eval.fa, reduce: last, name: fa_loss}
    lower_is_better: true
    seeds: [0, 1, 2]
    baseline: fixed
    variants:
      fixed:            [mixing.adaptive=false]
      bandit_per_token: [mixing.adaptive=true, mixing.fertility_aware=false]
      bandit_fertility: [mixing.adaptive=true, mixing.fertility_aware=true]

Each run gets ``--set seed=<s> --set output_dir=experiments/runs/<name>/<variant>/s<s>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from myllm.config import REPO_ROOT, get, load_config
from myllm.research import registry
from myllm.research.stats import bootstrap_ci, seed_summary


def read_metric(spec: dict, output_dir: str) -> float:
    path = Path(spec["file"].format(output_dir=output_dir))
    if path.suffix == ".jsonl":
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = lines if spec.get("reduce", "last") == "all" else [lines[-1]]
    else:
        rows = [json.loads(path.read_text(encoding="utf-8"))]
    vals = []
    for row in rows:
        node = row
        for part in spec["key"].split("."):
            node = node[part]
        vals.append(float(node))
    return sum(vals) / len(vals)


def run_ablation(cfg: dict, dry_run: bool = False) -> dict:
    name = cfg["name"]
    reg_path = REPO_ROOT / cfg.get("registry", "experiments/registry.jsonl")
    runs_root = cfg.get("runs_dir", "experiments/runs")
    cmd = [sys.executable if c == "python" else c for c in cfg["command"]]
    results: dict[str, list[float]] = {}
    for variant, overrides in cfg["variants"].items():
        for seed in cfg.get("seeds", [0]):
            out = f"{runs_root}/{name}/{variant}/s{seed}"
            full = [*cmd, "--set", f"seed={seed}", "--set", f"output_dir={out}"]
            for o in overrides:
                full += ["--set", o]
            print("$", " ".join(full), flush=True)
            if dry_run:
                continue
            subprocess.run(full, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
            value = read_metric(cfg["metric"], str(REPO_ROOT / out))
            results.setdefault(variant, []).append(value)
            registry.record(name, variant, seed, cfg["metric"].get("name", cfg["metric"]["key"]), value, overrides, out,
                            path=reg_path)
    return results


def report(cfg: dict, results: dict[str, list[float]]) -> str:
    lower = cfg.get("lower_is_better", False)
    base = cfg.get("baseline") or next(iter(results))
    metric = cfg["metric"].get("name", cfg["metric"]["key"])
    lines = [f"# Ablation: {cfg['name']}", "", f"metric: `{metric}` ({'lower' if lower else 'higher'} is better); "
             f"baseline: `{base}`; seeds: {cfg.get('seeds', [0])}", "",
             "| variant | mean | std | n | Δ vs baseline | 95% CI of Δ (bootstrap over seeds) |", "|---|---|---|---|---|---|"]
    b = results.get(base, [])
    for variant, vals in results.items():
        s = seed_summary(vals)
        if variant == base or not b:
            delta, ci = "—", "—"
        else:
            diffs = [v - bv for v, bv in zip(vals, b)]
            mean, lo, hi = bootstrap_ci(diffs)
            delta, ci = f"{mean:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"| {variant} | {s['mean']:.4f} | {s['std']:.4f} | {s['n']} | {delta} | {ci} |")
    lines += ["", "> A variant only 'wins' if the whole CI is on the better side of 0 — and with few seeds,"
              " treat any result as preliminary. Record negative results too."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="run an ablation matrix")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    results = run_ablation(cfg, a.dry_run)
    if a.dry_run:
        return 0
    md = report(cfg, results)
    out = REPO_ROOT / cfg.get("report_dir", "experiments/reports") / f"{cfg['name']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"report -> {out}")
    return 0 if get(cfg, "name") else 1


if __name__ == "__main__":
    raise SystemExit(main())
