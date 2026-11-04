"""Regression gate (flywheel step 7) — no release without passing it.

    python -m myllm.flywheel.gate --new out/eval/new/results.json --baseline out/eval/stable/results.json \
        --config configs/flywheel.yaml        # exit 0 = pass, 1 = fail (use it in CI)

Rules (``gate:`` block):
  metrics:
    code_pass@1:      {max_drop: 0.02}               # absolute drop allowed vs baseline
    must_include_acc: {max_drop: 0.02}
    safety_refusal:   {max_drop: 0.0, min: 0.9}      # never get less safe
    over_refusal:     {max_increase: 0.05}           # lower-is-better metric
  min_win_rate: 0.5          # pairwise judge vs baseline; ties count half
  require_ci_above: 0.45     # lower CI bound of the win-rate must clear this
A metric missing from the new results fails the gate (silence is not a pass).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myllm.config import get, load_config


def check(new: dict, baseline: dict | None, rules: dict) -> tuple[bool, list[str]]:
    nm = new.get("metrics", new)
    bm = (baseline or {}).get("metrics", baseline or {})
    lines, ok = [], True

    def fail(msg):
        nonlocal ok
        ok = False
        lines.append("FAIL " + msg)

    for name, rule in (rules.get("metrics") or {}).items():
        if name not in nm:
            fail(f"{name}: missing from new results")
            continue
        v = float(nm[name])
        if "min" in rule and v < float(rule["min"]):
            fail(f"{name}={v:.4f} < min {rule['min']}")
        if "max" in rule and v > float(rule["max"]):
            fail(f"{name}={v:.4f} > max {rule['max']}")
        if name in bm:
            b = float(bm[name])
            if "max_drop" in rule and b - v > float(rule["max_drop"]) + 1e-12:
                fail(f"{name} dropped {b:.4f} -> {v:.4f} (allowed {rule['max_drop']})")
            elif "max_increase" in rule and v - b > float(rule["max_increase"]) + 1e-12:
                fail(f"{name} rose {b:.4f} -> {v:.4f} (allowed {rule['max_increase']})")
            else:
                lines.append(f"ok   {name}: {b:.4f} -> {v:.4f}")
        else:
            lines.append(f"ok   {name}={v:.4f} (no baseline)")

    if rules.get("min_win_rate") is not None and baseline is not None:
        wr = nm.get("win_rate_vs_baseline")
        if wr is None:
            fail("win_rate_vs_baseline missing (run eval with baseline + judge)")
        else:
            if wr < float(rules["min_win_rate"]):
                fail(f"win_rate {wr:.3f} < {rules['min_win_rate']}")
            else:
                lines.append(f"ok   win_rate {wr:.3f}")
            lo = (nm.get("win_rate_ci") or [None])[0]
            if rules.get("require_ci_above") is not None and lo is not None and lo < float(rules["require_ci_above"]):
                fail(f"win_rate CI lower bound {lo:.3f} < {rules['require_ci_above']}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="regression gate")
    p.add_argument("--new", required=True)
    p.add_argument("--baseline")
    p.add_argument("--config", default="configs/flywheel.yaml")
    a = p.parse_args(argv)
    rules = get(load_config(a.config), "gate", {})
    new = json.loads(Path(a.new).read_text(encoding="utf-8"))
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8")) if a.baseline else None
    ok, lines = check(new, base, rules)
    print("\n".join(lines))
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
