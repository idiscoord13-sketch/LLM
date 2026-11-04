"""Minimal OpenAI-compatible chat client (vLLM, SGLang, Ollama, llama.cpp server, TGI...).

Kept dependency-light (stdlib ``urllib``) so data/eval scripts run anywhere.
Every teacher/judge/student endpoint in this repo goes through ``ChatClient``,
which makes swapping a 4060-hosted Ollama model for a cloud vLLM a config change.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _expand(value: str | None) -> str | None:
    """Expand ${ENV_VARS}; an unset variable becomes None instead of the literal '${...}'."""
    if not isinstance(value, str):
        return value
    out = os.path.expandvars(value)
    return None if "${" in out or (out.startswith("$") and len(out) > 1) else out


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    default_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, block: dict) -> ChatClient:
        return cls(
            base_url=(_expand(block["base_url"]) or "http://localhost:8000/v1").rstrip("/"),
            model=_expand(block["model"]),
            api_key=_expand(block.get("api_key")) or None,
            timeout=float(block.get("timeout", 120)),
            default_params=dict(block.get("params", {})),
        )

    def chat(self, messages: list[dict], n: int = 1, **params) -> list[str]:
        body = {"model": self.model, "messages": messages, "n": n, **self.default_params, **params}
        data = self._post("/chat/completions", body)
        return [c["message"].get("content") or "" for c in data["choices"]]

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (429, 500, 502, 503, 504)
                if attempt == self.max_retries or not retryable:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")


class FunctionClient:
    """Wrap a Python callable as a client — used for tests and local HF generation."""

    def __init__(self, fn: Callable[[list[dict], int], list[str]], model: str = "fn"):
        self.fn = fn
        self.model = model

    def chat(self, messages: list[dict], n: int = 1, **_params) -> list[str]:
        return self.fn(messages, n)


class HFGenerateClient:
    """Generate with a local transformers checkpoint (optionally + LoRA adapter)."""

    def __init__(self, model_name: str, adapter: str | None = None, max_new_tokens: int = 512,
                 template_kwargs: dict | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from myllm.tiny_models import resolve_model_name

        name = resolve_model_name(model_name)
        self.tok = AutoTokenizer.from_pretrained(adapter or name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = template_kwargs or {}
        self.model_name = model_name

    def chat(self, messages: list[dict], n: int = 1, temperature: float = 0.7, **params) -> list[str]:
        import torch

        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **self.template_kwargs
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=params.get("max_tokens", self.max_new_tokens),
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=params.get("top_p", 0.95) if temperature > 0 else None,
                num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [self.tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def make_client(block: dict):
    """``{type: openai|hf, ...}`` -> client."""
    kind = block.get("type", "openai")
    if kind == "openai":
        return ChatClient.from_config(block)
    if kind == "hf":
        return HFGenerateClient(block["model"], adapter=block.get("adapter"),
                                max_new_tokens=int(block.get("max_new_tokens", 512)),
                                template_kwargs=block.get("chat_template_kwargs"))
    raise ValueError(f"unknown client type {kind!r}")
