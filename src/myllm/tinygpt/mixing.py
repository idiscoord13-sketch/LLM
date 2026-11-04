"""Idea #4 — Fertility-Aware Adaptive Data Mixing (a bandit over data domains).

Fixed mixing ratios (e.g. 50% fa / 40% en / 10% code) are a guess. Here the ratio
adapts during pre-training: every ``update_every`` steps we measure each domain's
held-out loss and reward domains whose loss is dropping fastest *per token spent*.

Fertility-aware part: losses are compared in **bits per character**, not per token
(loss_per_token / chars_per_token / ln 2). Persian usually needs more tokens per
word than English, so a per-token comparison is biased; per-character improvement
is what users actually feel. Ablation: ``fertility_aware: false`` uses per-token loss.

Close prior work: DoReMi (Xie et al. 2023) and online data mixing (Albalak et al.
2023). The fertility term and the low-resource-language focus are the additions;
novelty is modest — the value is practical (see docs/research/idea-4.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MixingBandit:
    domains: list[str]
    prior: list[float]
    chars_per_token: dict[str, float]
    eta: float = 2.0  # how aggressively to follow rewards
    min_prob: float = 0.05  # exploration floor: never starve a domain ("cheap but needed")
    fertility_aware: bool = True
    probs: list[float] = field(default_factory=list)
    _last_metric: dict[str, float] = field(default_factory=dict)
    _scores: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        total = sum(self.prior)
        self.prior = [p / total for p in self.prior]
        self.probs = list(self.prior)
        self._scores = {d: 0.0 for d in self.domains}

    def metric(self, domain: str, loss_per_token: float) -> float:
        if self.fertility_aware:
            return loss_per_token / max(1e-6, self.chars_per_token.get(domain, 1.0)) / math.log(2)
        return loss_per_token

    def update(self, eval_losses: dict[str, float], tokens_seen: dict[str, int]) -> list[float]:
        rewards = {}
        for d in self.domains:
            m = self.metric(d, eval_losses[d])
            prev = self._last_metric.get(d)
            spent = max(1, tokens_seen.get(d, 0))
            rewards[d] = 0.0 if prev is None else (prev - m) / spent * 1e6  # improvement per M tokens
            self._last_metric[d] = m
        if all(v == 0.0 for v in rewards.values()):
            return self.probs
        # normalise rewards to unit scale, then exponentiated-gradient update around the prior
        scale = max(abs(v) for v in rewards.values()) or 1.0
        for d in self.domains:
            self._scores[d] = 0.7 * self._scores[d] + 0.3 * rewards[d] / scale
        logits = [math.log(p) + self.eta * self._scores[d] for p, d in zip(self.prior, self.domains)]
        mx = max(logits)
        raw = [math.exp(x - mx) for x in logits]
        z = sum(raw)
        probs = [r / z for r in raw]
        k = len(probs)
        floor = min(self.min_prob, 1.0 / k)
        probs = [floor + (1 - k * floor) * p for p in probs]
        self.probs = probs
        self.history.append({"rewards": rewards, "probs": dict(zip(self.domains, probs))})
        return probs
