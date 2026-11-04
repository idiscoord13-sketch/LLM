"""Defensive filters for user data (flywheel step 4, roadmap §7.4).

* consent          — already enforced by the store; re-checked here (defence in depth)
* prompt injection — user text that tries to *instruct the future model* is dropped
* toxicity         — keyword floor + pluggable classifier hook
* per-user cap     — no single user may contribute more than ``max_user_share`` of a batch
                     (the main defence against data poisoning by one account)
* dedup            — near-identical prompts collapse to one
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable

from myllm.data.dedup import exact_hash

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior) ",
    r"you are now\b",
    r"from now on,? (you|always)",
    r"system prompt",
    r"jailbreak",
    r"\bDAN\b",
    r"از این به بعد (همیشه|تو)",
    r"دستور(ات|های)? قبلی را (نادیده|فراموش)",
    r"(نقش|وانمود کن).{0,20}(بدون محدودیت|هیچ قانونی)",
    r"پرامپت سیستم",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Minimal floor; plug a real classifier via `toxicity_fn` (e.g. a multilingual toxicity model).
TOXIC_KEYWORDS = ["kill yourself", "خودت را بکش", "بمب بساز", "how to make a bomb"]


def has_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def is_toxic(text: str, toxicity_fn: Callable[[str], float] | None = None, threshold: float = 0.5) -> bool:
    low = (text or "").lower()
    if any(k in low for k in TOXIC_KEYWORDS):
        return True
    return bool(toxicity_fn and toxicity_fn(text) >= threshold)


def cap_per_user(rows: list[dict], max_share: float = 0.2, seed: int = 0) -> list[dict]:
    if not rows:
        return rows
    limit = max(1, int(max_share * len(rows)))
    by_user: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_user[r.get("user_id") or "anon"].append(r)
    rng = random.Random(seed)
    out = []
    for _user, items in by_user.items():
        if len(items) > limit:
            items = rng.sample(items, limit)
        out.extend(items)
    return out


def apply_filters(rows: list[dict], cfg: dict | None = None,
                  toxicity_fn: Callable[[str], float] | None = None) -> tuple[list[dict], dict]:
    cfg = cfg or {}
    stats: Counter = Counter(input=len(rows))
    kept, seen = [], set()
    for r in rows:
        if not r.get("consent"):
            stats["no_consent"] += 1
            continue
        user_text = "\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
        all_text = user_text + "\n" + (r.get("feedback", {}).get("edit") or "")
        if has_injection(all_text):
            stats["prompt_injection"] += 1
            continue
        if is_toxic(all_text, toxicity_fn):
            stats["toxic"] += 1
            continue
        h = exact_hash(user_text + "||" + r["messages"][-1]["content"])
        if h in seen:
            stats["duplicate"] += 1
            continue
        seen.add(h)
        kept.append(r)
    before = len(kept)
    kept = cap_per_user(kept, float(cfg.get("max_user_share", 0.2)), int(cfg.get("seed", 0)))
    stats["user_cap"] += before - len(kept)
    stats["output"] = len(kept)
    return kept, dict(stats)
