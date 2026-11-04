"""KTO from binary feedback (👍/👎) — the natural loss for the data flywheel (roadmap §6.3, §7.3).

    python -m myllm.train.kto --config configs/kto.yaml --profile local_4060
    python -m myllm.train.kto --config configs/smoke/kto.yaml --smoke

Rows::

    {"prompt": [{"role": "user", ...}], "completion": [{"role": "assistant", ...}],
     "label": true, "confidence": 0.9}

Idea #2 — Confidence-Gated KTO: every label carries a ``confidence`` (user
reliability x signal agreement x judge agreement, computed in
``myllm.flywheel.signals``). Rows below ``kto.min_confidence`` are *not* trained on;
they are written to a human-review queue. The rest are sampled proportionally to
their confidence (``WeightedSamplerMixin``) so noisy labels have less pull.
Set ``kto.confidence_weighting: false`` for the plain-KTO baseline in ablations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from myllm.config import add_common_args, get, load_config
from myllm.train.common import (
    finish,
    load_rows,
    log,
    prepare_run,
    setup_logging,
    weighted_trainer_cls,
)
from myllm.train.dpo import load_policy
from myllm.train.modeling import load_tokenizer, peft_config_for, training_kwargs
from myllm.utils import write_jsonl


def gate_by_confidence(rows: list[dict], min_conf: float) -> tuple[list[dict], list[dict]]:
    keep, review = [], []
    for r in rows:
        (keep if float(r.get("confidence", 1.0)) >= min_conf else review).append(r)
    return keep, review


def balanced_class_weights(rows: list[dict]) -> tuple[float, float]:
    """KTO paper: desirable_w * n_desirable / (undesirable_w * n_undesirable) should be in [1, 4/3]."""
    n_pos = sum(1 for r in rows if r["label"])
    n_neg = len(rows) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    # fix undesirable weight to 1, choose desirable weight so the ratio is ~1.0-1.33
    return max(1.0, n_neg / n_pos), 1.0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = add_common_args(argparse.ArgumentParser(description="KTO (confidence-gated optional)"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    out = prepare_run(cfg)

    from datasets import Dataset
    from trl import KTOConfig, KTOTrainer

    rows = load_rows(cfg, smoke=args.smoke)
    min_conf = float(get(cfg, "kto.min_confidence", 0.0))
    rows, review = gate_by_confidence(rows, min_conf)
    if review:
        write_jsonl(Path(out) / "review_queue.jsonl", review)
        log.info("confidence gate: %d rows sent to human review (< %.2f)", len(review), min_conf)

    use_conf = get(cfg, "kto.confidence_weighting", True)
    weights = [float(r.get("confidence", 1.0)) for r in rows] if use_conf else None
    if weights and len(set(weights)) == 1:
        weights = None  # all equal -> plain sampler
    dw, uw = balanced_class_weights(rows)

    tokenizer = load_tokenizer(cfg)
    dataset = Dataset.from_list([{k: r[k] for k in ("prompt", "completion", "label")} for r in rows])
    model = load_policy(cfg)
    targs = KTOConfig(
        **training_kwargs(cfg, smoke=args.smoke),
        beta=float(get(cfg, "preference.beta", 0.1)),
        desirable_weight=float(get(cfg, "kto.desirable_weight", dw)),
        undesirable_weight=float(get(cfg, "kto.undesirable_weight", uw)),
        max_length=get(cfg, "train.max_length", 2048),
    )
    trainer_cls = weighted_trainer_cls(KTOTrainer) if weights else KTOTrainer
    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for(model, cfg),
    )
    if weights:
        trainer.sample_weights = weights
    trainer.train()
    finish(trainer, tokenizer, cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
