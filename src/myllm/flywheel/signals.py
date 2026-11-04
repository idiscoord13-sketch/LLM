"""Quality signals + per-label confidence (flywheel step 3; idea #2).

Signals per conversation (last assistant turn):
  * thumb up / down
  * regenerate          -> implicit dissatisfaction
  * user edit           -> the golden signal: the answer the user actually wanted

Confidence of a binary label = product of three cheap estimates, each in [0, 1]:
  1. user reliability   — how often this user's votes agree with the consensus on
                          prompts that several users rated (Beta-smoothed toward a prior)
  2. signal agreement   — do the signals agree? (👍 + regenerate is contradictory)
  3. judge agreement    — optional: does an LLM judge agree with the label?
"""

from __future__ import annotations

from collections import defaultdict

from myllm.data.dedup import canonical


def last_turn(row: dict) -> tuple[list[dict], str]:
    msgs = row["messages"]
    idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    return msgs[:idx], msgs[idx]["content"]


def label_of(feedback: dict) -> bool | None:
    """True = desirable, False = undesirable, None = no usable signal."""
    if feedback.get("thumb") == "up" and not feedback.get("regenerate"):
        return True
    if feedback.get("thumb") == "down" or feedback.get("regenerate") or feedback.get("edit"):
        return False  # an edit means the original answer was not what the user wanted
    return None


def signal_agreement(feedback: dict) -> float:
    up, down = feedback.get("thumb") == "up", feedback.get("thumb") == "down"
    regen, edit = bool(feedback.get("regenerate")), bool(feedback.get("edit"))
    if up and (regen or edit):
        return 0.3  # contradictory
    if down and (regen or edit):
        return 1.0  # several signals agree it was bad
    if up or down:
        return 0.8
    if regen or edit:
        return 0.7
    return 0.0


def user_reliability(rows: list[dict], prior: float = 0.7, strength: float = 4.0) -> dict[str, float]:
    """Agreement with the per-prompt majority, Beta-smoothed: (agree + prior*s) / (n + s)."""
    votes: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for r in rows:
        lab = label_of(r.get("feedback", {}))
        if lab is None or not r.get("user_id"):
            continue
        prompt, _ = last_turn(r)
        votes[canonical(prompt[-1]["content"]) if prompt else ""].append((r["user_id"], lab))
    agree: dict[str, float] = defaultdict(float)
    total: dict[str, float] = defaultdict(float)
    for vs in votes.values():
        if len({u for u, _ in vs}) < 2:
            continue  # need at least two users to define a consensus
        pos = sum(1 for _, lab in vs if lab)
        majority = pos * 2 >= len(vs)
        for u, lab in vs:
            total[u] += 1
            agree[u] += float(lab == majority)
    users = {r.get("user_id") for r in rows if r.get("user_id")}
    return {u: (agree[u] + prior * strength) / (total[u] + strength) for u in users}


def confidence(row: dict, reliability: dict[str, float], judge_agrees: bool | None = None,
               default_reliability: float = 0.7) -> float:
    c = reliability.get(row.get("user_id"), default_reliability) * signal_agreement(row.get("feedback", {}))
    if judge_agrees is not None:
        c *= 1.0 if judge_agrees else 0.4
    return round(c, 4)
