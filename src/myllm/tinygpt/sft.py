"""Supervised fine-tuning of the from-scratch GPT on chat data (Phase 1, last step).

    python -m myllm.tinygpt.sft --config configs/tiny_gpt.yaml --set sft.data=data/samples/sft_fa.jsonl

Same rules as the big-model SFT: ChatML template via ``apply_chat_template`` and
loss only on assistant tokens (``myllm.data.chat_format``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myllm.config import add_common_args, get, load_config, resolve_path
from myllm.data.chat_format import IGNORE_INDEX, tokenize_conversation
from myllm.tinygpt.model import GPT, GPTConfig
from myllm.utils import read_jsonl, set_seed


def load_checkpoint(path: str | Path, device: str = "cpu"):
    from transformers import AutoTokenizer

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer"])
    return model, tokenizer, ckpt


def collate(examples: list[dict], pad_id: int, device: str):
    n = max(len(e["input_ids"]) for e in examples)
    x = torch.full((len(examples), n - 1), pad_id, dtype=torch.long)
    y = torch.full((len(examples), n - 1), IGNORE_INDEX, dtype=torch.long)
    for i, e in enumerate(examples):
        ids, labels = e["input_ids"], e["labels"]
        x[i, : len(ids) - 1] = torch.tensor(ids[:-1])
        y[i, : len(ids) - 1] = torch.tensor(labels[1:])  # predict token t+1 from prefix <= t
    return x.to(device), y.to(device)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="SFT the tiny GPT"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    set_seed(int(get(cfg, "seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(get(cfg, "output_dir", "out/tinygpt"))
    model, tok, ckpt = load_checkpoint(out / "ckpt.pt", device)

    rows = read_jsonl(resolve_path(get(cfg, "sft.data", "data/samples/sft_fa.jsonl")))
    examples = [tokenize_conversation(tok, r["messages"], model.cfg.max_seq_len) for r in rows]
    examples = [e for e in examples if any(lab != IGNORE_INDEX for lab in e["labels"])]
    opt = model.configure_optimizer(float(get(cfg, "sft.learning_rate", 1e-4)), 0.0, device=device)
    steps = 10 if args.smoke else int(get(cfg, "sft.steps", 500))
    bs = int(get(cfg, "sft.batch_size", 8))
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        idx = torch.randint(0, len(examples), (min(bs, len(examples)),), generator=g).tolist()
        x, y = collate([examples[i] for i in idx], tok.pad_token_id, device)
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"sft step {step} loss {loss.item():.4f}")
    torch.save({**ckpt, "model": model.state_dict(), "sft_steps": steps}, out / "ckpt_sft.pt")
    print(f"saved {out / 'ckpt_sft.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
