"""Model merging without training (roadmap §6.5): LoRA merge, linear, SLERP, TIES, DARE.

    # 1) bake a LoRA adapter into its base model
    python -m myllm.train.merge lora --base Qwen/Qwen3-4B-Instruct-2507 --adapter out/sft --out out/sft-merged
    # 2) combine fine-tunes of the same base (e.g. a Persian-strong and a code-strong student)
    python -m myllm.train.merge ties --base Qwen/Qwen3-4B-Instruct-2507 \
        --models out/fa-merged out/code-merged --weights 0.5 0.5 --density 0.5 --out out/merged

Task vector of a fine-tune = finetuned - base. TIES (Yadav et al. 2023): trim each task
vector to its top-``density`` magnitudes, elect a sign per parameter by total mass,
average only the agreeing entries. DARE (Yu et al. 2023): randomly drop ``1-density``
of each task vector and rescale by ``1/density`` before merging.

The tensor math is implemented here (and unit-tested) so you can see exactly what
happens; for 14B+ models prefer ``mergekit`` which streams tensors from disk.
"""

from __future__ import annotations

import argparse

import torch


def task_vector(finetuned: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return finetuned.float() - base.float()


def trim_topk(delta: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return delta
    k = max(1, int(delta.numel() * density))
    threshold = delta.abs().flatten().kthvalue(delta.numel() - k + 1).values
    return torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))


def dare(delta: torch.Tensor, density: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if density >= 1.0:
        return delta
    keep = torch.rand(delta.shape, generator=generator) < density
    return torch.where(keep, delta / density, torch.zeros_like(delta))


def ties_merge(deltas: list[torch.Tensor], weights: list[float], density: float = 0.5,
               sparsify: str = "magnitude", generator: torch.Generator | None = None) -> torch.Tensor:
    trimmed = []
    for d, w in zip(deltas, weights):
        d = trim_topk(d, density) if sparsify == "magnitude" else dare(d, density, generator)
        trimmed.append(d * w)
    stacked = torch.stack(trimmed)
    sign = torch.sign(stacked.sum(dim=0))  # elect sign by total mass
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    summed = (stacked * agree).sum(dim=0)
    count = (agree.float() * torch.tensor(weights, dtype=torch.float32).abs().view(-1, *[1] * (stacked.dim() - 1))).sum(0)
    return torch.where(count > 0, summed / count.clamp_min(1e-8), torch.zeros_like(summed))


def linear_merge(deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(weights)
    return sum(d * (w / total) for d, w in zip(deltas, weights))


def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    a32, b32 = a.float().flatten(), b.float().flatten()
    an, bn = a32 / (a32.norm() + eps), b32 / (b32.norm() + eps)
    dot = torch.clamp((an * bn).sum(), -1.0, 1.0)
    if 1 - dot.abs() < 1e-6:  # nearly parallel -> lerp
        out = (1 - t) * a32 + t * b32
    else:
        theta = torch.arccos(dot)
        out = (torch.sin((1 - t) * theta) * a32 + torch.sin(t * theta) * b32) / torch.sin(theta)
    return out.view_as(a).to(a.dtype)


def merge_state_dicts(base: dict, models: list[dict], method: str, weights: list[float],
                      density: float = 0.5, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, base_t in base.items():
        if not torch.is_floating_point(base_t):
            out[name] = base_t
            continue
        if method == "slerp":
            assert len(models) == 2, "slerp merges exactly two models"
            out[name] = slerp(models[0][name], models[1][name], weights[1] / sum(weights))
            continue
        deltas = [task_vector(m[name], base_t) for m in models]
        if method == "linear":
            merged = linear_merge(deltas, weights)
        elif method == "ties":
            merged = ties_merge(deltas, weights, density, "magnitude")
        elif method == "dare_ties":
            merged = ties_merge(deltas, weights, density, "dare", gen)
        elif method == "dare_linear":
            merged = linear_merge([dare(d, density, gen) for d in deltas], weights)
        else:
            raise ValueError(f"unknown merge method {method}")
        out[name] = (base_t.float() + merged).to(base_t.dtype)
    return out


def merge_lora(base: str, adapter: str, out: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from myllm.tiny_models import resolve_model_name

    model = AutoModelForCausalLM.from_pretrained(resolve_model_name(base), dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="merge models")
    parser.add_argument("method", choices=["lora", "linear", "slerp", "ties", "dare_ties", "dare_linear"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.method == "lora":
        merge_lora(args.base, args.adapter, args.out)
        return 0
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from myllm.tiny_models import resolve_model_name

    load = lambda p: AutoModelForCausalLM.from_pretrained(resolve_model_name(p), dtype=torch.bfloat16)  # noqa: E731
    base_model = load(args.base)
    others = [load(p).state_dict() for p in args.models]
    weights = args.weights or [1.0] * len(others)
    merged = merge_state_dicts(base_model.state_dict(), others, args.method, weights, args.density, args.seed)
    base_model.load_state_dict(merged)
    base_model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(resolve_model_name(args.models[0])).save_pretrained(args.out)
    print(f"merged {len(others)} models with {args.method} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
