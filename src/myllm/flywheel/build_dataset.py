"""Turn filtered feedback into training data (flywheel step 5).

Outputs:
  * ``sft.jsonl``  — 👍 answers and user edits (edit = the answer the user wanted)
  * ``dpo.jsonl``  — pairs: (👍 vs 👎 on the same prompt) and idea #1 (user edit vs original)
  * ``kto.jsonl``  — every labelled answer with a confidence (idea #2)
Each file is mixed with a replay sample of older/general data (anti-forgetting).

Idea #1 — Edit-Signal Preference Mining: pair (edited answer = chosen, original =
rejected) only when the normalised edit distance d is inside [low, high]:
too small = typo fixes/noise, too large = the user rewrote for a different intent.
Pair weight = clip(d / saturation, w_min, 1) so substantive edits pull harder.
"""

from __future__ import annotations

import difflib
import random
from collections import defaultdict
from pathlib import Path

from myllm.config import resolve_path
from myllm.data.dedup import canonical
from myllm.flywheel.signals import confidence, label_of, last_turn, user_reliability
from myllm.utils import read_jsonl, write_jsonl


def edit_distance(a: str, b: str) -> float:
    """Normalised distance in [0, 1] (1 - difflib similarity ratio)."""
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def edit_pair(prompt: list[dict], original: str, edited: str, low: float = 0.05, high: float = 0.8,
              saturation: float = 0.3, w_min: float = 0.2) -> dict | None:
    d = edit_distance(original, edited)
    if not low <= d <= high:
        return None
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": edited}],
        "rejected": [{"role": "assistant", "content": original}],
        "weight": round(min(1.0, max(w_min, d / saturation)), 4),
        "source": "flywheel:edit",
        "edit_distance": round(d, 4),
    }


def build(rows: list[dict], cfg: dict | None = None) -> dict[str, list[dict]]:
    cfg = cfg or {}
    edit_cfg = cfg.get("edit_pairs", {})
    reliability = user_reliability(rows)
    sft, dpo, kto = [], [], []
    by_prompt: dict[str, dict[str, list]] = defaultdict(lambda: {"good": [], "bad": []})

    for r in rows:
        fb = r.get("feedback", {})
        prompt, answer = last_turn(r)
        if not prompt:
            continue
        lab = label_of(fb)
        conf = confidence(r, reliability)
        if fb.get("edit"):
            sft.append({"messages": [*prompt, {"role": "assistant", "content": fb["edit"]}],
                        "source": "flywheel:edit", "weight": 1.0})
            if edit_cfg.get("enabled", True):
                pair = edit_pair(prompt, answer, fb["edit"], **{k: v for k, v in edit_cfg.items() if k != "enabled"})
                if pair:
                    dpo.append(pair)
        elif lab is True:
            sft.append({"messages": [*prompt, {"role": "assistant", "content": answer}],
                        "source": "flywheel:thumb_up", "weight": conf})
        if lab is not None:
            kto.append({"prompt": prompt, "completion": [{"role": "assistant", "content": answer}],
                        "label": lab, "confidence": conf, "source": "flywheel"})
            key = canonical("\n".join(m["content"] for m in prompt))
            by_prompt[key]["good" if lab else "bad"].append((prompt, answer, conf))

    for groups in by_prompt.values():
        for (p, good, cg), (_, bad, cb) in zip(groups["good"], groups["bad"]):
            dpo.append({"prompt": p, "chosen": [{"role": "assistant", "content": good}],
                        "rejected": [{"role": "assistant", "content": bad}],
                        "weight": round(min(cg, cb) or 0.1, 4), "source": "flywheel:thumbs"})
    return {"sft": sft, "dpo": dpo, "kto": kto}


def mix_replay(new: list[dict], replay_files: list[str], ratio: float, seed: int = 0) -> list[dict]:
    """Add ``ratio * len(new)`` rows sampled from older data so the model keeps what it knew."""
    if not new or ratio <= 0 or not replay_files:
        return list(new)
    pool: list[dict] = []
    for f in replay_files:
        p = resolve_path(f)
        if Path(p).exists():
            pool.extend({**r, "source": r.get("source", "replay")} for r in read_jsonl(p))
    if not pool:
        return list(new)
    rng = random.Random(seed)
    k = min(len(pool), max(1, int(ratio * len(new))))
    out = list(new) + rng.sample(pool, k)
    rng.shuffle(out)
    return out


def write_all(datasets: dict[str, list[dict]], out_dir: str | Path, replay: dict | None = None) -> dict[str, int]:
    out_dir = Path(out_dir)
    counts = {}
    for name, rows in datasets.items():
        r = (replay or {}).get(name, {})
        rows = mix_replay(rows, r.get("files", []), float(r.get("ratio", 0.0)))
        counts[name] = write_jsonl(out_dir / f"{name}.jsonl", rows)
    return counts
