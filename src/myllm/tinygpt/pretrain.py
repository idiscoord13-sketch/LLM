"""Pre-train the from-scratch GPT (Phase 1).

    python -m myllm.tinygpt.pretrain --config configs/tiny_gpt.yaml           # RTX 4060, ~30M params
    python -m myllm.tinygpt.pretrain --config configs/smoke/tiny_gpt.yaml     # CPU, seconds

Steps:
  1. train a byte-level BPE (HF tokenizers) on the corpora, with the ChatML template
  2. tokenize every domain into a flat uint32 array (+ a held-out slice for eval)
  3. sample batches by domain probabilities (fixed, or the idea-#4 bandit)
  4. AdamW + warmup + cosine/WSD schedule, grad accumulation, bf16 autocast on GPU,
     grad clipping, periodic per-domain eval, checkpoints and a JSONL log
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from myllm.config import add_common_args, get, load_config, resolve_path
from myllm.tinygpt.mixing import MixingBandit
from myllm.tinygpt.model import GPT, GPTConfig
from myllm.tokenizer_utils import build_tokenizer, fertility
from myllm.utils import iter_jsonl, save_run_config, set_seed


def read_texts(path: str, field: str = "text", limit: int | None = None, domain: str | None = None) -> list[str]:
    """Texts from a JSONL (``field`` column, optionally only rows with ``domain``) or a .txt file."""
    p = resolve_path(path)
    texts: list[str] = []
    if p.suffix == ".jsonl":
        for row in iter_jsonl(p):
            if domain is not None and row.get("domain") != domain:
                continue
            if isinstance(row.get(field), str):
                texts.append(row[field])
            if limit and len(texts) >= limit:
                break
    else:
        texts = [t for t in p.read_text(encoding="utf-8").split("\n\n") if t.strip()][:limit]
    return texts


def lr_at(step: int, cfg: dict) -> float:
    """Warmup then cosine, or WSD (warmup-stable-decay: flat, then a short final decay)."""
    lr, min_lr = float(get(cfg, "train.learning_rate", 3e-4)), float(get(cfg, "train.min_lr", 3e-5))
    total, warmup = int(get(cfg, "train.max_steps")), int(get(cfg, "train.warmup_steps", 100))
    if step < warmup:
        return lr * (step + 1) / warmup
    schedule = get(cfg, "train.schedule", "cosine")
    if schedule == "wsd":
        decay_start = int(total * (1 - float(get(cfg, "train.decay_fraction", 0.2))))
        if step < decay_start:
            return lr
        frac = (step - decay_start) / max(1, total - decay_start)
        return min_lr + (lr - min_lr) * (1 - frac)
    frac = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, frac)))


class DomainData:
    """Per-domain flat token arrays with a held-out tail for evaluation."""

    def __init__(self, arrays: dict[str, np.ndarray], seq_len: int, holdout: float, seed: int):
        self.seq_len = seq_len
        self.train, self.val = {}, {}
        for name, arr in arrays.items():
            cut = max(seq_len + 2, int(len(arr) * (1 - holdout)))
            cut = min(cut, len(arr))
            self.train[name], self.val[name] = arr[:cut], arr[cut - (seq_len + 1):] if holdout > 0 else arr
            if len(self.val[name]) < seq_len + 1:
                self.val[name] = arr
        self.rng = np.random.default_rng(seed)

    def batch(self, split: str, domain: str, batch_size: int, device: str):
        arr = (self.train if split == "train" else self.val)[domain]
        if len(arr) <= self.seq_len + 1:
            reps = math.ceil((self.seq_len + 2) / max(1, len(arr)))
            arr = np.tile(arr, reps)
        starts = self.rng.integers(0, len(arr) - self.seq_len - 1, size=batch_size)
        x = np.stack([arr[s : s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([arr[s + 1 : s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def mixed_batch(self, probs: list[float], domains: list[str], batch_size: int, device: str):
        picks = self.rng.choice(len(domains), size=batch_size, p=np.asarray(probs) / sum(probs))
        xs, ys, counts = [], [], {d: 0 for d in domains}
        for i in picks:
            x, y = self.batch("train", domains[i], 1, device)
            xs.append(x)
            ys.append(y)
            counts[domains[i]] += self.seq_len
        return torch.cat(xs), torch.cat(ys), counts


@torch.no_grad()
def evaluate(model, data: DomainData, domains: list[str], batches: int, batch_size: int, device: str, ctx):
    model.eval()
    out = {}
    for d in domains:
        losses = []
        for _ in range(batches):
            x, y = data.batch("val", d, batch_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[d] = float(np.mean(losses))
    model.train()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="pre-train the tiny GPT"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)
    seed = int(get(cfg, "seed", 42))
    set_seed(seed)
    out = Path(get(cfg, "output_dir", "out/tinygpt"))
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data + tokenizer
    sources = get(cfg, "data.sources")
    limit = 200 if args.smoke else get(cfg, "data.max_docs_per_source")
    texts = {s["name"]: read_texts(s["path"], s.get("field", "text"), limit, s.get("domain")) for s in sources}
    tok_dir = out / "tokenizer"
    if (tok_dir / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        all_texts = [t for ts in texts.values() for t in ts]
        tokenizer = build_tokenizer(all_texts, tok_dir, vocab_size=int(get(cfg, "tokenizer.vocab_size", 16384)),
                                    min_frequency=int(get(cfg, "tokenizer.min_frequency", 2)))
    eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    arrays, cpt = {}, {}
    for name, ts in texts.items():
        ids: list[int] = []
        for t in ts:
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
            ids.append(eos)
        arrays[name] = np.asarray(ids, dtype=np.uint32)
        cpt[name] = fertility(tokenizer, ts[:500])["chars_per_token"]
        print(f"domain {name}: {len(ts)} docs, {len(ids):,} tokens, chars/token={cpt[name]:.2f}")

    seq_len = int(get(cfg, "model.max_seq_len", 512))
    data = DomainData(arrays, seq_len, float(get(cfg, "data.holdout", 0.02)), seed)
    domains = [s["name"] for s in sources]
    prior = [float(s.get("weight", 1.0)) for s in sources]

    # 2) model
    mcfg = GPTConfig(vocab_size=len(tokenizer), **{k: v for k, v in get(cfg, "model").items() if k != "vocab_size"})
    model = GPT(mcfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M params ({model.num_params(True) / 1e6:.2f}M non-embedding)")
    opt = model.configure_optimizer(float(get(cfg, "train.learning_rate", 3e-4)),
                                    float(get(cfg, "train.weight_decay", 0.1)), device=device)
    if get(cfg, "train.compile", False) and device == "cuda":
        model = torch.compile(model)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    # 3) mixing
    bandit = None
    if get(cfg, "mixing.adaptive", False):
        bandit = MixingBandit(domains, prior, cpt, eta=float(get(cfg, "mixing.eta", 2.0)),
                              min_prob=float(get(cfg, "mixing.min_prob", 0.05)),
                              fertility_aware=bool(get(cfg, "mixing.fertility_aware", True)))
    probs = bandit.probs if bandit else [p / sum(prior) for p in prior]

    # 4) train
    max_steps = 30 if args.smoke else int(get(cfg, "train.max_steps"))
    cfg.setdefault("train", {})["max_steps"] = max_steps
    bs, accum = int(get(cfg, "train.batch_size", 32)), int(get(cfg, "train.grad_accum", 1))
    eval_every = 10 if args.smoke else int(get(cfg, "train.eval_every", 250))
    save_run_config(out, cfg, extra={"params": model.num_params() if hasattr(model, "num_params") else None})
    log_f = open(out / "log.jsonl", "a", encoding="utf-8")
    tokens_since = {d: 0 for d in domains}
    t0 = time.time()
    first_loss = last_loss = None
    for step in range(max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        loss_acc = 0.0
        for _ in range(accum):
            x, y, counts = data.mixed_batch(probs, domains, bs, device)
            for d, c in counts.items():
                tokens_since[d] += c
            with ctx:
                _, loss, _ = model(x, y)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(get(cfg, "train.grad_clip", 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        first_loss = loss_acc if first_loss is None else first_loss
        last_loss = loss_acc

        if step % eval_every == 0 or step == max_steps - 1:
            ev = evaluate(model, data, domains, 2 if args.smoke else 10, bs, device, ctx)
            if bandit:
                probs = bandit.update(ev, tokens_since)
                tokens_since = {d: 0 for d in domains}
            elapsed = time.time() - t0
            rec = {"step": step, "loss": round(loss_acc, 4), "eval": {k: round(v, 4) for k, v in ev.items()},
                   "lr": opt.param_groups[0]["lr"], "probs": dict(zip(domains, [round(p, 3) for p in probs])),
                   "tok_per_s": round((step + 1) * accum * bs * seq_len / max(1e-6, elapsed))}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(rec)

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict(), "config": mcfg.to_dict(), "tokenizer": str(tok_dir), "step": max_steps},
               out / "ckpt.pt")
    log_f.close()
    print(f"done: loss {first_loss:.3f} -> {last_loss:.3f}; checkpoint {out / 'ckpt.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
