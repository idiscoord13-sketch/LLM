"""A modern decoder-only Transformer written from scratch (roadmap §5.3–5.4, Phase 1).

Every component of today's open LLMs (Llama/Qwen/Gemma family), in ~200 lines:

  * RMSNorm (pre-norm)                       — simpler & more stable than LayerNorm
  * RoPE rotary position embeddings          — relative positions, extendable context
  * GQA grouped-query attention              — n_kv_head < n_head -> smaller KV cache
  * SwiGLU feed-forward                      — gated MLP used by every modern LLM
  * QK-norm (optional)                       — RMSNorm on q/k, as in Qwen3; stabilizes training
  * tied input/output embeddings             — fewer parameters for small models
  * KV cache for fast autoregressive generation

Read it top to bottom; each block is independent and unit-tested in tests/test_tinygpt.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2  # GQA: 3 query heads share each key/value head
    d_model: int = 384
    ffn_mult: float = 8 / 3  # SwiGLU hidden = mult * d_model (rounded), keeps params ~= 4x MLP
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """y = x / rms(x) * g  — no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_frequencies(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D). Rotates each (pair of) channels by a position-dependent angle."""
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head (GQA groups)"
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # GQA: repeat each kv head for its group of query heads
        groups = self.n_head // self.n_kv_head
        k_rep = k.repeat_interleave(groups, dim=1)
        v_rep = v.repeat_interleave(groups, dim=1)

        if kv_cache is None:
            y = F.scaled_dot_product_attention(
                q, k_rep, v_rep, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # new queries may attend to every cached position and causally among themselves
            S = k_rep.shape[2]
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y), new_cache


class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * (W_up x) )."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        hidden = 64 * math.ceil(hidden / 64)  # hardware-friendly multiple
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        h, cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_frequencies(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 / nanoGPT trick)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
        start_pos: int = 0,
    ):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.embed(idx)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, None if kv_caches is None else kv_caches[i])
            new_caches.append(cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss, new_caches

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95), device: str = "cpu"):
        """AdamW; no weight decay on norms/biases/1-D params (standard practice)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Sample with a KV cache: the prompt is processed once, then one token per step."""
        self.eval()
        idx = idx[:, -self.cfg.max_seq_len :]
        logits, _, caches = self(idx)
        pos = idx.shape[1]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = next_logits.argmax(-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k:
                    kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    probs = sorted_logits.softmax(-1)
                    remove = probs.cumsum(-1) - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                nxt = torch.multinomial(next_logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if pos >= self.cfg.max_seq_len:
                break
            logits, _, caches = self(nxt, kv_caches=caches, start_pos=pos)
            pos += 1
        return idx


def estimate_flops_per_token(cfg: GPTConfig) -> float:
    """~6N FLOPs per training token (forward + backward), the C ≈ 6·N·D rule of roadmap §1.2."""
    model = GPT(cfg)
    return 6.0 * model.num_params(exclude_embeddings=False)
