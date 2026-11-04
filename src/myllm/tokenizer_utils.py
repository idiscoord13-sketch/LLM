"""Tokenizer helpers: train a byte-level BPE, wrap it as an HF tokenizer with a
ChatML chat template (the template family Qwen uses), and measure fertility.

Used by Phase 1 (our own tokenizer) and by the offline test-suite (a tiny
tokenizer that behaves like Qwen's: <|im_start|>role\\n ... <|im_end|>).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# ChatML. Kept deliberately close to Qwen's so masking/template tests transfer.
CHATML_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 16384,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
):
    """Train a byte-level BPE with HF `tokenizers` (Rust, fast). Returns a `tokenizers.Tokenizer`.

    Byte-level means no <unk>: any Persian/English/code/emoji string is representable.
    """
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        trainers,
    )

    special_tokens = special_tokens or SPECIAL_TOKENS
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def wrap_hf(tok, chat_template: str = CHATML_TEMPLATE):
    """Wrap a `tokenizers.Tokenizer` as a transformers tokenizer with a chat template."""
    from transformers import PreTrainedTokenizerFast

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=[t for t in SPECIAL_TOKENS if t not in ("<|im_end|>", "<|endoftext|>")],
    )
    hf.chat_template = chat_template
    return hf


def build_tokenizer(texts: Iterable[str], save_dir: str | Path, vocab_size: int = 16384, min_frequency: int = 2):
    hf = wrap_hf(train_bpe(texts, vocab_size=vocab_size, min_frequency=min_frequency))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(save_dir)
    return hf


_WORD = re.compile(r"[\w‌]+", re.UNICODE)


def fertility(tokenizer, texts: Iterable[str]) -> dict[str, float]:
    """Tokens per word (lower is better) and chars per token (higher is better)."""
    n_tokens = n_words = n_chars = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens += len(ids)
        n_words += len(_WORD.findall(text))
        n_chars += len(text)
    return {
        "tokens": float(n_tokens),
        "words": float(n_words),
        "tokens_per_word": n_tokens / max(1, n_words),
        "chars_per_token": n_chars / max(1, n_tokens),
    }
