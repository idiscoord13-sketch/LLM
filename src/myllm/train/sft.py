"""Supervised fine-tuning with (Q)LoRA (roadmap §6.2, Phase 2).

    python -m myllm.train.sft --config configs/sft.yaml --profile local_4060 --smoke
    python -m myllm.train.sft --config configs/sft.yaml --profile local_4060
    python -m myllm.train.sft --config configs/smoke/sft.yaml          # offline CPU check

Data: JSONL of ``{"messages": [...]}`` (or Alpaca/ShareGPT — auto-converted), optional
``weight`` per row (verifier-weighted distillation, idea #7).

The dataset is tokenized here with ``apply_chat_template`` and assistant-only labels
(see ``myllm.data.chat_format``), then handed to TRL's SFTTrainer pre-tokenized,
so the loss mask does not depend on the TRL version or on ``{% generation %}`` tags.
"""

from __future__ import annotations

import argparse

from myllm.config import add_common_args, get, load_config
from myllm.data.chat_format import tokenize_conversation
from myllm.data.formats import to_messages, validate_messages
from myllm.train.common import (
    extract_weights,
    finish,
    load_rows,
    log,
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


def build_dataset(rows: list[dict], tokenizer, max_length: int, template_kwargs: dict | None = None):
    from datasets import Dataset

    records, weights, stats = [], [], {"invalid": 0, "no_target": 0, "truncated": 0, "fallback": 0}
    for row in rows:
        conv = to_messages(row)
        if validate_messages(conv["messages"]):
            stats["invalid"] += 1
            continue
        enc = tokenize_conversation(tokenizer, conv["messages"], max_length, template_kwargs)
        if all(lab == -100 for lab in enc["labels"]):
            stats["no_target"] += 1  # e.g. assistant turn truncated away
            continue
        stats["truncated"] += int(enc["info"]["truncated"])
        stats["fallback"] += int(enc["info"]["fallback_last_turn"])
        records.append({"input_ids": enc["input_ids"], "labels": enc["labels"]})
        weights.append(float(row.get("weight", 1.0)))
    log.info("SFT dataset: %d examples, skipped/flags=%s", len(records), stats)
    if not records:
        raise ValueError("no usable SFT examples")
    return Dataset.from_list(records), weights, stats


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = add_common_args(argparse.ArgumentParser(description="SFT with (Q)LoRA"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    prepare_run(cfg)

    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(cfg)
    rows = load_rows(cfg, smoke=args.smoke)
    max_length = get(cfg, "train.max_length", 2048)
    dataset, weights, _ = build_dataset(rows, tokenizer, max_length, get(cfg, "chat_template_kwargs"))

    model = load_model(cfg)
    sft_args = SFTConfig(
        **training_kwargs(cfg, smoke=args.smoke),
        max_length=max_length,
        packing=get(cfg, "train.packing", False),
        padding_free=get(cfg, "train.padding_free", False),
    )
    trainer_cls = weighted_trainer_cls(SFTTrainer) if extract_weights(rows) else SFTTrainer
    trainer = trainer_cls(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for(model, cfg),
    )
    if trainer_cls is not SFTTrainer:
        trainer.sample_weights = weights
    trainer.train()
    finish(trainer, tokenizer, cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
