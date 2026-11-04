"""RLVR with GRPO — reinforcement learning from verifiable rewards (roadmap §6.4).

    python -m myllm.train.grpo --config configs/grpo.yaml --profile cloud_24gb
    python -m myllm.train.grpo --config configs/smoke/grpo.yaml --smoke

For each prompt the policy samples a *group* of ``num_generations`` answers; each
answer's advantage is its reward minus the group mean (no value network, no learned
reward model). Rewards here are verifiable: unit tests actually executed
(``myllm.eval.code_exec``), plus small shaping terms. Do SFT (+DPO) first — RL only
sharpens abilities the model already has at least occasionally.

Rows: ``{"prompt": [{"role": "user", ...}], "tests": "assert f(1) == 2\\n..."}``
"""

from __future__ import annotations

import argparse

from myllm.config import add_common_args, get, load_config
from myllm.train.common import finish, load_rows, prepare_run, setup_logging
from myllm.train.dpo import load_policy
from myllm.train.modeling import load_tokenizer, peft_config_for, training_kwargs
from myllm.train.rewards import REWARDS


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = add_common_args(argparse.ArgumentParser(description="GRPO with verifiable rewards"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    prepare_run(cfg)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = load_tokenizer(cfg)
    rows = load_rows(cfg, smoke=args.smoke)
    dataset = Dataset.from_list([{"prompt": r["prompt"], "tests": r.get("tests", "")} for r in rows])
    reward_cfg = get(cfg, "grpo.rewards", {"code_tests": 1.0, "code_format": 0.2})
    funcs = [REWARDS[name] for name in reward_cfg]
    weights = [float(w) for w in reward_cfg.values()]

    t = training_kwargs(cfg, smoke=args.smoke)
    n_gen = int(get(cfg, "grpo.num_generations", 4))
    # generation batch must be divisible by num_generations
    t["per_device_train_batch_size"] = max(t["per_device_train_batch_size"], n_gen)
    t["remove_unused_columns"] = False
    targs = GRPOConfig(
        **t,
        num_generations=n_gen,
        max_completion_length=int(get(cfg, "grpo.max_completion_length", 512)),
        temperature=float(get(cfg, "grpo.temperature", 1.0)),
        beta=float(get(cfg, "grpo.beta", 0.0)),  # KL to reference; 0 = no ref model (saves VRAM)
        loss_type=get(cfg, "grpo.loss_type", "dapo"),
        epsilon_high=float(get(cfg, "grpo.epsilon_high", 0.28)),  # DAPO clip-higher
        mask_truncated_completions=get(cfg, "grpo.mask_truncated_completions", True),
        reward_weights=weights,
        use_vllm=get(cfg, "grpo.use_vllm", False),
        chat_template_kwargs=get(cfg, "chat_template_kwargs") or None,
        log_completions=get(cfg, "grpo.log_completions", False),
    )
    model = load_policy(cfg)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=funcs,
        args=targs,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config_for(model, cfg),
    )
    trainer.train()
    finish(trainer, tokenizer, cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
