"""Sample from the tiny GPT.

    python -m myllm.tinygpt.generate --ckpt out/tinygpt/ckpt.pt --text "زبان فارسی"
    python -m myllm.tinygpt.generate --ckpt out/tinygpt/ckpt_sft.pt --chat "سلام! خودت را معرفی کن."
"""

from __future__ import annotations

import argparse

import torch

from myllm.tinygpt.sft import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text", help="raw continuation prompt")
    p.add_argument("--chat", help="user message (uses the chat template)")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_checkpoint(args.ckpt, device)
    if args.chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": args.chat}], tokenize=False,
                                         add_generation_prompt=True)
    else:
        prompt = args.text or ""
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eos_id=tok.eos_token_id)
    print(tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
