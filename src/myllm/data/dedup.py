"""Exact + near-duplicate removal (roadmap §4.2 step 3).

* exact: SHA-1 of a canonical form (lowercased, whitespace collapsed, ZWNJ removed)
* fuzzy: MinHash + LSH over character 5-gram shingles (works for Persian, English and code)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from datasketch import MinHash, MinHashLSH

_WS = re.compile(r"\s+")


def canonical(text: str) -> str:
    return _WS.sub(" ", text.replace("‌", "")).strip().lower()


def exact_hash(text: str) -> str:
    return hashlib.sha1(canonical(text).encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    t = canonical(text)
    if len(t) <= n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def minhash(text: str, num_perm: int = 128, n: int = 5) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    m.update_batch([s.encode("utf-8") for s in shingles(text, n)])
    return m


def dedup_exact(texts: Iterable[str]) -> list[int]:
    """Return indices of the first occurrence of each exact-duplicate group."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, text in enumerate(texts):
        h = exact_hash(text)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def dedup_minhash(
    texts: list[str], threshold: float = 0.8, num_perm: int = 128, n: int = 5
) -> list[int]:
    """Greedy near-dup removal: keep a text unless it collides with an already kept one."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[int] = []
    for i, text in enumerate(texts):
        m = minhash(text, num_perm=num_perm, n=n)
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(i)
    return keep


def jaccard(a: str, b: str, n: int = 5) -> float:
    sa, sb = shingles(a, n), shingles(b, n)
    return len(sa & sb) / max(1, len(sa | sb))
