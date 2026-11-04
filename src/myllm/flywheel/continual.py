"""Idea #3 — Replay-Anchored Continual LoRA.

Weekly retraining invites catastrophic forgetting and drift. The recipe:

1. **Replay buffer** — every week's SFT/preference data is mixed with a sample of
   (a) earlier weeks and (b) "anchor" data the deployed model already handles well
   (prompts it passed on the golden/regression sets). See ``build_replay_buffer``.
2. **KL anchor** — preference training uses the *currently deployed* model as the
   DPO/KTO reference, so ``beta`` directly penalises drifting away from it
   (``anchor_overrides`` sets ``model.adapter``/reference accordingly).
3. **Periodic consolidation** — every N weeks the weekly adapters are merged with
   TIES into a new base (``consolidate``) instead of stacking adapters forever.
4. **Regression gate** before any release (``myllm.flywheel.gate``).

Ablations to run (docs/research/idea-3.md): no replay / no anchor / no merge.
"""

from __future__ import annotations

import random
from pathlib import Path

from myllm.utils import read_jsonl, write_jsonl


def build_replay_buffer(history_files: list[str], anchor_rows: list[dict], size: int, anchor_share: float = 0.5,
                        seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    history: list[dict] = []
    for f in history_files:
        if Path(f).exists():
            history.extend(read_jsonl(f))
    n_anchor = min(len(anchor_rows), int(size * anchor_share))
    n_hist = min(len(history), size - n_anchor)
    buf = rng.sample(anchor_rows, n_anchor) + rng.sample(history, n_hist)
    rng.shuffle(buf)
    return [{**r, "source": r.get("source", "replay")} for r in buf]


def anchor_rows_from_eval(per_item: list[dict], golden: list[dict], answers: dict[str, str]) -> list[dict]:
    """Golden items the deployed model got right become SFT anchors (its own correct answers)."""
    gold = {g["id"]: g for g in golden}
    rows = []
    for it in per_item:
        good = it.get("code_pass") == 1.0 or it.get("must_include") == 1.0 or it.get("judge") == "A"
        if good and it["id"] in gold and it["id"] in answers:
            rows.append({"messages": [{"role": "user", "content": gold[it["id"]]["prompt"]},
                                      {"role": "assistant", "content": answers[it["id"]]}], "source": "anchor"})
    return rows


def anchor_overrides(deployed_adapter: str | None, beta: float = 0.1) -> list[str]:
    """CLI overrides that make the deployed model the starting point and KL reference."""
    out = [f"preference.beta={beta}"]
    if deployed_adapter:
        out.append(f"model.adapter={deployed_adapter}")
    return out


def should_consolidate(week: int, every: int = 4) -> bool:
    return week > 0 and week % every == 0


def consolidate_cmd(base: str, merged_weekly_models: list[str], out: str, density: float = 0.5) -> list[str]:
    """Command line for TIES-merging the (LoRA-merged) weekly models into a new base."""
    return ["python", "-m", "myllm.train.merge", "ties", "--base", base, "--models", *merged_weekly_models,
            "--density", str(density), "--out", out]


def save_buffer(rows: list[dict], path: str) -> int:
    return write_jsonl(path, rows)
