"""Byte-level BPE from scratch — the educational version (Phase 1).

This is the algorithm GPT-2/Llama/Qwen tokenizers use, in ~80 readable lines:

1. pre-tokenize text into chunks (words, numbers, punctuation, spaces) so merges
   never cross word boundaries;
2. start from the 256 bytes (so any UTF-8 string — Persian included — is encodable);
3. repeatedly merge the most frequent adjacent pair into a new token.

It is slow (pure Python). For real vocabularies use ``myllm.tokenizer_utils.train_bpe``
(HF `tokenizers`, Rust) — it implements the same idea at scale. Compare their
fertility on Persian with ``scripts/fertility.py``.
"""

from __future__ import annotations

import re
from collections import Counter

# words (any script, incl. ZWNJ inside Persian words), digits one by one, punctuation runs, whitespace
PRETOKENIZE = re.compile(r"\s?[^\W\d_](?:[^\W\d_]|‌)*|\d|\s?[^\w\s]+|\s+(?!\S)|\s+", re.UNICODE)


class ByteBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _pair_counts(chunks: Counter) -> Counter:
        counts: Counter = Counter()
        for ids, freq in chunks.items():
            for pair in zip(ids, ids[1:]):
                counts[pair] += freq
        return counts

    @staticmethod
    def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    def train(self, texts: list[str], vocab_size: int, verbose: bool = False) -> ByteBPE:
        assert vocab_size > 256
        chunks: Counter = Counter()
        for text in texts:
            for piece in PRETOKENIZE.findall(text):
                chunks[tuple(piece.encode("utf-8"))] += 1
        for new_id in range(256, vocab_size):
            counts = self._pair_counts(chunks)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            chunks = Counter({self._merge(ids, pair, new_id): f for ids, f in chunks.items()})
            if verbose:
                print(new_id, repr(self.vocab[new_id].decode("utf-8", "replace")), freq)
        return self

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in PRETOKENIZE.findall(text):
            ids = tuple(piece.encode("utf-8"))
            while len(ids) >= 2:
                # apply the earliest-learned merge available (same order as training)
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = self._merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
