"""Read (and optionally download) the public Ganjoor export (github.com/ganjoor/ganjoor-data).

Each poem file carries the metre label (``Metre.Rhythm``, e.g.
"مفاعیلن مفاعیلن مفاعیلن مفاعیلن (هزج مثمن سالم)") and its hemistichs, many of them
partially vowelised (اَدِرْ، بویِ). That gives us, for free, a large labelled set to
**evaluate** the scanner and to **learn** a word-weight lexicon (``myllm.aruz.lexicon``).

License: at the time of writing the ganjoor-data repo has no LICENSE file. The classical
poems are public domain, but the compilation/vowelisation is Ganjoor's editorial work —
keep downloads local (``data/raw/ganjoor`` is git-ignored) and ask Ganjoor before
redistributing or selling anything derived from their files.

    python -m myllm.aruz.ganjoor download --cat /hafez/ghazal --max-poems 500
    python -m myllm.aruz.ganjoor download --cat /saadi/divan/ghazals --max-poems 300
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/ganjoor/ganjoor-data/master/"
DEFAULT_DIR = Path("data/raw/ganjoor")


@dataclass
class Hemistich:
    poem_url: str
    couplet: int
    position: str  # Right = first hemistich (مصراع اول), Left = second
    text: str
    rhythm: str | None  # raw Ganjoor label, e.g. "فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف)"
    poet: str


def _fetch_json(path: str, base: str = RAW_BASE, retries: int = 3):
    url = base + path.lstrip("/")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return None


_KEEP = ("Id", "CatId", "Title", "FullUrl", "RhymeLetters", "Metre", "LiteraryTags", "Tags")
_KEEP_VERSE = ("VOrder", "Position", "Text", "CoupletIndex")


def slim(poem: dict) -> dict:
    """Keep what scanning/evaluation needs (drops long summaries and HTML -> ~5x smaller on disk)."""
    out = {k: poem[k] for k in _KEEP if k in poem}
    out["Verses"] = [{k: v[k] for k in _KEEP_VERSE if k in v} for v in poem.get("Verses", [])]
    return out


def list_poems(cat_url: str, max_poems: int = 500, recursive: bool = True) -> list[str]:
    """Breadth-first walk of a category tree; returns up to ``max_poems`` poem urls."""
    urls: list[str] = []
    queue = [cat_url]
    while queue and len(urls) < max_poems:
        cat = _fetch_json(f"poets{queue.pop(0)}/_cat.json")
        urls.extend(p["FullUrl"] for p in cat.get("Poems", []))
        if recursive:
            queue.extend(child["FullUrl"] for child in cat.get("ChildCats", []))
    return urls[:max_poems]


def download(cat_url: str, out_dir: Path = DEFAULT_DIR, max_poems: int = 500, recursive: bool = True,
             threads: int = 8) -> int:
    """Download the poems of a category (e.g. ``/hafez/ghazal``) into ``out_dir`` (parallel, resumable)."""
    from concurrent.futures import ThreadPoolExecutor

    urls = list_poems(cat_url, max_poems, recursive)

    def one(url: str) -> bool:
        dest = Path(out_dir) / (url.strip("/") + ".json")
        if dest.exists():
            return False
        try:
            poem = _fetch_json(f"poets{url}.json")
        except Exception:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps(slim(poem), ensure_ascii=False), encoding="utf-8")
        tmp.replace(dest)  # never leave a half-written poem behind
        return True

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, urls))
    return len(urls)


def iter_poem_files(root: Path = DEFAULT_DIR) -> Iterator[Path]:
    yield from sorted(Path(root).rglob("*.json"))


def hemistichs(poem: dict) -> list[Hemistich]:
    rhythm = (poem.get("Metre") or {}).get("Rhythm")
    poet = poem.get("FullUrl", "/?").split("/")[1]
    out = []
    for v in poem.get("Verses", []):
        if v.get("Position") not in ("Right", "Left"):
            continue  # skip band/paragraph/single lines of mixed forms
        text = (v.get("Text") or "").strip()
        if text:
            out.append(Hemistich(poem.get("FullUrl", ""), v.get("CoupletIndex", -1), v["Position"], text, rhythm, poet))
    return out


def load_corpus(root: Path = DEFAULT_DIR, require_metre: bool = True) -> list[Hemistich]:
    rows: list[Hemistich] = []
    for f in iter_poem_files(root):
        poem = json.loads(f.read_text(encoding="utf-8"))
        hs = hemistichs(poem)
        if require_metre and not (hs and hs[0].rhythm):
            continue
        rows.extend(hs)
    return rows


_PAREN = re.compile(r"\((.*?)\)")


def split_rhythm(rhythm: str) -> tuple[str, str | None]:
    """'مفاعیلن ×4 (هزج مثمن سالم)' -> ('مفاعیلن ...', 'هزج مثمن سالم')."""
    m = _PAREN.search(rhythm or "")
    name = m.group(1).strip() if m else None
    feet = _PAREN.sub("", rhythm or "").strip()
    return feet, name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ganjoor export helper")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--cat", required=True, help="category url, e.g. /hafez/ghazal")
    d.add_argument("--out", default=str(DEFAULT_DIR))
    d.add_argument("--max-poems", type=int, default=500)
    s = sub.add_parser("stats")
    s.add_argument("--root", default=str(DEFAULT_DIR))
    a = p.parse_args(argv)
    if a.cmd == "download":
        print(f"downloaded {download(a.cat, Path(a.out), a.max_poems)} poems from {a.cat}")
    else:
        from collections import Counter

        rows = load_corpus(Path(a.root))
        print(f"{len(rows)} hemistichs")
        for r, c in Counter(h.rhythm for h in rows).most_common(30):
            print(f"{c:6d}  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
