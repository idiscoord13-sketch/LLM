"""On-policy distillation (GKD) — the strongest lever for small students.

    python -m myllm.train.gkd --config configs/gkd.yaml --profile cloud_24gb
    python -m myllm.train.gkd --config configs/smoke/gkd.yaml --smoke

The student *generates* answers to the prompts and is trained to match the
teacher's next-token distribution on its own samples (generalized JSD; ``beta``
interpolates forward KL (0) ... reverse KL (1)). Unlike plain SFT on teacher text,
this fixes the student's own mistakes (no exposure bias), which is why on-policy
distillation is so effective for 1-4B models (Agarwal et al. 2023, arXiv 2306.13649;
TRL ``DistillationTrainer``).

Requirements: teacher and student must share a tokenizer/vocabulary (e.g. Qwen3-32B
teacher -> Qwen3-4B student). The teacher must fit next to the student; on an 8 GB
4060 use offline distillation (``myllm.distill.teacher_gen`` + SFT) instead and run
GKD on a rented GPU.

Rows: ``{"prompt": [{"role": "user", "content": ...}]}``
"""

from __future__ import annotations

import argparse

from myllm.config import add_common_args, get, load_config
from myllm.tiny_models import resolve_model_name
from myllm.train.common import finish, load_rows, prepare_run, setup_logging
from myllm.train.dpo import load_policy
from myllm.train.modeling import (
    load_tokenizer,
    model_init_kwargs,
    peft_config_for,
    training_kwargs,
)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = add_common_args(argparse.ArgumentParser(description="On-policy distillation (GKD)"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    prepare_run(cfg)

    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import DistillationConfig, DistillationTrainer

    tokenizer = load_tokenizer(cfg)
    rows = load_rows(cfg, smoke=args.smoke)
    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])
    student = load_policy(cfg)
    teacher_name = resolve_model_name(get(cfg, "model.teacher"))
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, **model_init_kwargs(cfg))
    teacher.eval()

    t = training_kwargs(cfg, smoke=args.smoke)
    t["remove_unused_columns"] = False  # the trainer needs the raw "prompt" column
    targs = DistillationConfig(
        **t,
        beta=float(get(cfg, "gkd.beta", 0.5)),
        temperature=float(get(cfg, "gkd.temperature", 0.9)),
        max_completion_length=get(cfg, "gkd.max_completion_length", 512),
        chat_template_kwargs=get(cfg, "chat_template_kwargs") or None,
    )
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=targs,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for(student, cfg),
    )
    trainer.train()
    finish(trainer, tokenizer, cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
