"""Roadmap mistakes #2 and #3: wrong chat template / loss on the prompt. These tests guard both."""

import pytest

from myllm.data.chat_format import IGNORE_INDEX, render_text, tokenize_conversation


def _learned_text(tok, enc):
    return tok.decode([i for i, lab in zip(enc["input_ids"], enc["labels"]) if lab != IGNORE_INDEX])


def test_template_is_chatml(tok):
    text = render_text(tok, [{"role": "user", "content": "سلام"}], add_generation_prompt=True)
    assert text == "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n"


def test_only_assistant_tokens_are_learned(tok):
    msgs = [{"role": "system", "content": "دستیار باش"}, {"role": "user", "content": "پایتخت ایران؟"},
            {"role": "assistant", "content": "تهران"}]
    enc = tokenize_conversation(tok, msgs)
    assert _learned_text(tok, enc) == "تهران<|im_end|>\n"  # includes end-of-turn so the model learns to stop
    assert enc["labels"][0] == IGNORE_INDEX
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])


def test_multi_turn_masks_every_assistant_turn(tok):
    msgs = [{"role": "user", "content": "یک"}, {"role": "assistant", "content": "دو"},
            {"role": "user", "content": "سه"}, {"role": "assistant", "content": "چهار"}]
    enc = tokenize_conversation(tok, msgs)
    assert enc["info"]["assistant_turns"] == 2
    learned = _learned_text(tok, enc)
    assert "دو" in learned and "چهار" in learned and "سه" not in learned and "user" not in learned


def test_truncation_flag(tok):
    msgs = [{"role": "user", "content": "سلام " * 50}, {"role": "assistant", "content": "درود"}]
    enc = tokenize_conversation(tok, msgs, max_length=16)
    assert enc["info"]["truncated"] and len(enc["input_ids"]) == 16


def test_history_rewriting_template_falls_back_to_last_turn(tok):
    """Qwen3-style templates drop <think> from earlier turns; prefixes then stop matching."""
    import copy

    t2 = copy.deepcopy(tok)
    t2.chat_template = (
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'assistant' and not loop.last -%}"
        "{{- '<|im_start|>assistant\\n' + (m['content'].split('</think>')[-1]) + '<|im_end|>\\n' -}}"
        "{%- else -%}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' -}}{%- endif -%}"
        "{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
    )
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>x</think>b"},
            {"role": "user", "content": "c"}, {"role": "assistant", "content": "<think>y</think>d"}]
    enc = tokenize_conversation(t2, msgs)
    assert enc["info"]["fallback_last_turn"]
    assert _learned_text(t2, enc).startswith("<think>y</think>d")


def test_sft_build_dataset_skips_bad_rows(tok):
    pytest.importorskip("datasets")
    from myllm.train.sft import build_dataset

    rows = [{"messages": [{"role": "user", "content": "سلام"}, {"role": "assistant", "content": "درود"}]},
            {"messages": [{"role": "assistant", "content": "بدون کاربر"}]},
            {"instruction": "جمع ۲ و ۲", "output": "۴"}]
    ds, weights, stats = build_dataset(rows, tok, max_length=64)
    assert len(ds) == 2 and stats["invalid"] == 1 and weights == [1.0, 1.0]
    assert set(ds.column_names) == {"input_ids", "labels"}
