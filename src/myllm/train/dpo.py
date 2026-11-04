"""Preference optimization: DPO and friends (roadmap §6.3, Phase 3).

    python -m myllm.train.dpo --config configs/dpo.yaml --profile local_4060
    python -m myllm.train.dpo --config configs/smoke/dpo.yaml --smoke

``preference.algorithm``:
  * ``dpo``      — sigmoid DPO (default; most documented)
  * ``ipo``      — less overfitting on small preference sets
  * ``apo_zero`` — anchored preference optimization
  * ``robust``   — DPO robust to label noise (good for user feedback)
  * ``simpo``    — reference-free SimPO (via TRL's CPOTrainer with cpo_alpha=0)
  * ``mix``      — e.g. DPO + SFT term: set ``preference.loss_type: [sigmoid, sft]``

Data rows (conversational)::

    {"prompt": [{"role": "user", ...}], "chosen": [{"role": "assistant", ...}],
     "rejected": [{"role": "assistant", ...}], "weight": 1.0}

``weight`` is optional (idea #1: edit-distance-weighted pairs from the flywheel).
Start from the SFT adapter: set ``model.name`` to the merged SFT model or keep the
base and set ``model.adapter`` to the SFT LoRA directory.
"""

from __future__ import annotations

import argparse

from myllm.config import add_common_args, get, load_config
from myllm.train.common import (
    extract_weights,
    finish,
    load_rows,
    prepare_run,
    setup_logging,
    weighted_trainer_cls,
)
from myllm.train.modeling import (
    load_model,
    load_tokenizer,
    peft_config_for,
    training_kwargs,
)

DPO_LOSSES = {"dpo": "sigmoid", "ipo": "ipo", "apo_zero": "apo_zero", "robust": "robust", "hinge": "hinge"}


def _rows_to_dataset(rows: list[dict]):
    from datasets import Dataset

    keep = ("prompt", "chosen", "rejected")
    return Dataset.from_list([{k: r[k] for k in keep} for r in rows])


def load_policy(cfg: dict):
    """Base model, optionally with an existing (SFT) LoRA adapter loaded as trainable."""
    model = load_model(cfg)
    adapter = get(cfg, "model.adapter")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        model._myllm_peft_applied = True
    return model


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = add_common_args(argparse.ArgumentParser(description="DPO / IPO / SimPO preference optimization"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    prepare_run(cfg)

    tokenizer = load_tokenizer(cfg)
    rows = load_rows(cfg, smoke=args.smoke)
    dataset = _rows_to_dataset(rows)
    weights = extract_weights(rows)
    model = load_policy(cfg)
    algo = get(cfg, "preference.algorithm", "dpo")
    common = dict(
        **training_kwargs(cfg, smoke=args.smoke),
        beta=float(get(cfg, "preference.beta", 0.1)),
        max_length=get(cfg, "train.max_length", 2048),
    )

    if algo == "simpo":
        from trl.experimental.cpo import CPOConfig, CPOTrainer

        targs = CPOConfig(
            **common,
            loss_type="simpo",
            cpo_alpha=float(get(cfg, "preference.cpo_alpha", 0.0)),
            simpo_gamma=float(get(cfg, "preference.simpo_gamma", 0.5)),
        )
        base_cls = CPOTrainer
    else:
        from trl import DPOConfig, DPOTrainer

        loss_type = get(cfg, "preference.loss_type") or DPO_LOSSES[algo]
        targs = DPOConfig(
            **common,
            loss_type=loss_type,
            loss_weights=get(cfg, "preference.loss_weights"),
            label_smoothing=float(get(cfg, "preference.label_smoothing", 0.0)),
            use_weighting=get(cfg, "preference.wpo_weighting", False),
        )
        base_cls = DPOTrainer

    trainer_cls = weighted_trainer_cls(base_cls) if weights else base_cls
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
