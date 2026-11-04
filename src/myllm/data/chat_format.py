"""Chat templating + assistant-only loss masking (roadmap §4.4, §6.2; mistakes #2 and #3).

Rules enforced here:
  * the prompt is ALWAYS rendered with ``tokenizer.apply_chat_template`` — never by hand;
  * labels are -100 everywhere except the tokens the assistant produced
    (including the end-of-turn token, so the model learns to stop).

How the mask is built (version-independent, works with any template — including
Qwen3's, which has no ``{% generation %}`` markers that TRL's built-in
``assistant_only_loss`` needs):

    full   = render(messages)
    for every assistant turn i:
        start = len(render(messages[:i], add_generation_prompt=True))
        end   = len(render(messages[:i+1]))
        labels[start:end] = full[start:end]

If a template rewrites history (e.g. Qwen3 drops <think> blocks of earlier turns),
prefixes stop matching; we then fall back to training on the final assistant turn
only and report it in the returned ``info``.
"""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


def _render_ids(tokenizer, messages: list[dict], add_generation_prompt: bool = False, **kw) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw
    )
    # transformers >= 5 may return a BatchEncoding/dict; older versions return a list.
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def render_text(tokenizer, messages: list[dict], add_generation_prompt: bool = False, **kw) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw
    )


def tokenize_conversation(
    tokenizer,
    messages: list[dict],
    max_length: int | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``input_ids``, ``labels``, ``attention_mask``, ``assistant_masks`` and ``info``."""
    kw = template_kwargs or {}
    full = _render_ids(tokenizer, messages, **kw)
    labels = [IGNORE_INDEX] * len(full)
    info = {"fallback_last_turn": False, "truncated": False, "assistant_turns": 0}

    spans: list[tuple[int, int]] = []
    consistent = True
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = _render_ids(tokenizer, messages[:i], add_generation_prompt=True, **kw)
        upto = _render_ids(tokenizer, messages[: i + 1], **kw)
        if full[: len(prefix)] != prefix or full[: len(upto)] != upto or len(upto) <= len(prefix):
            consistent = False
            break
        spans.append((len(prefix), len(upto)))

    if not consistent:
        info["fallback_last_turn"] = True
        last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
        prefix = _render_ids(tokenizer, messages[:last], add_generation_prompt=True, **kw)
        full = _render_ids(tokenizer, messages[: last + 1], **kw)
        if full[: len(prefix)] != prefix:
            raise ValueError("chat template is not prefix-consistent even for the last turn")
        labels = [IGNORE_INDEX] * len(full)
        spans = [(len(prefix), len(full))]

    for start, end in spans:
        labels[start:end] = full[start:end]
    info["assistant_turns"] = len(spans)

    if max_length is not None and len(full) > max_length:
        full, labels = full[:max_length], labels[:max_length]
        info["truncated"] = True

    assistant_masks = [0 if lab == IGNORE_INDEX else 1 for lab in labels]
    return {
        "input_ids": full,
        "labels": labels,
        "attention_mask": [1] * len(full),
        "assistant_masks": assistant_masks,
        "info": info,
    }


def masked_fraction(labels: list[int]) -> float:
    return sum(1 for x in labels if x == IGNORE_INDEX) / max(1, len(labels))
