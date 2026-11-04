"""Data cleaning pipeline CLI (roadmap §4.2, in the correct order).

    normalize -> langID -> exact dedup -> MinHash dedup -> quality -> PII -> decontam

Works on two kinds of JSONL rows:
  * raw text for (continued) pre-training: ``{"text": "..."}``
  * conversations for SFT: anything ``formats.to_messages`` understands

Usage:
    python -m myllm.data.pipeline --config configs/clean.yaml
    python -m myllm.data.pipeline --config configs/smoke/clean.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from myllm.config import add_common_args, get, load_config, resolve_path
from myllm.data.decontam import Decontaminator
from myllm.data.dedup import dedup_exact, dedup_minhash
from myllm.data.formats import conversation_text, to_messages, validate_messages
from myllm.data.langid import guess_language
from myllm.data.licenses import allowed as license_allowed
from myllm.data.normalize_fa import normalize_preserving_code
from myllm.data.pii import scrub_pii
from myllm.data.quality import heuristic_quality
from myllm.utils import iter_jsonl, read_jsonl, write_jsonl


def _row_text(row: dict) -> str:
    return row["text"] if "text" in row else conversation_text(row["messages"])


def _map_text(row: dict, fn) -> dict:
    if "text" in row:
        return {**row, "text": fn(row["text"])}
    return {**row, "messages": [{**m, "content": fn(m["content"])} for m in row["messages"]]}


def load_eval_texts(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for p in paths:
        for row in iter_jsonl(resolve_path(p)):
            for key in ("prompt", "reference", "question", "answer"):
                if isinstance(row.get(key), str):
                    texts.append(row[key])
    return texts


def run_pipeline(rows: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    steps = get(cfg, "steps", {}) or {}
    stats: dict = {"input": len(rows), "dropped": Counter(), "pii": Counter(), "lang": Counter()}

    # 0. license gate + format: conversations -> messages, validate
    commercial_safe = bool(get(cfg, "commercial_safe", True))
    allow_unknown = bool(get(cfg, "allow_unknown_license", True))
    out: list[dict] = []
    for row in rows:
        if not license_allowed(row.get("license"), commercial_safe, allow_unknown):
            stats["dropped"]["license"] += 1
            continue
        if "text" not in row:
            try:
                row = to_messages(row, system=get(cfg, "system_prompt"))
            except ValueError:
                stats["dropped"]["bad_format"] += 1
                continue
            if validate_messages(row["messages"]):
                stats["dropped"]["invalid_conversation"] += 1
                continue
        out.append(row)
    rows = out

    # 1. normalization (code blocks untouched)
    if steps.get("normalize", True):
        norm_kw = get(cfg, "normalize", {}) or {}
        rows = [_map_text(r, lambda t: normalize_preserving_code(t, **norm_kw)) for r in rows]

    # 2. language ID
    allowed = set(get(cfg, "langid.allowed", ["fa", "en", "code", "mixed"]))
    if steps.get("langid", True):
        kept = []
        for r in rows:
            lang = guess_language(_row_text(r)).lang
            stats["lang"][lang] += 1
            if lang in allowed:
                kept.append({**r, "lang": lang})
            else:
                stats["dropped"][f"lang_{lang}"] += 1
        rows = kept

    # 3. dedup: exact then fuzzy
    if steps.get("dedup_exact", True):
        keep = dedup_exact(_row_text(r) for r in rows)
        stats["dropped"]["exact_dup"] += len(rows) - len(keep)
        rows = [rows[i] for i in keep]
    if steps.get("dedup_minhash", True):
        keep = dedup_minhash(
            [_row_text(r) for r in rows],
            threshold=get(cfg, "dedup.threshold", 0.8),
            num_perm=get(cfg, "dedup.num_perm", 128),
        )
        stats["dropped"]["near_dup"] += len(rows) - len(keep)
        rows = [rows[i] for i in keep]

    # 4. quality
    if steps.get("quality", True):
        q_kw = get(cfg, "quality", {}) or {}
        kept = []
        for r in rows:
            report = heuristic_quality(_row_text(r), is_code=r.get("lang") == "code", **q_kw)
            if report.keep:
                kept.append(r)
            else:
                for reason in report.reasons:
                    stats["dropped"][f"quality_{reason}"] += 1
        rows = kept

    # 5. PII — mandatory before any training data is written
    if steps.get("pii", True):
        scrubbed = []
        for r in rows:
            counts: Counter = Counter()

            def _scrub(t: str, _c: Counter = counts) -> str:
                clean, c = scrub_pii(t)
                _c.update(c)
                return clean

            scrubbed.append(_map_text(r, _scrub))
            stats["pii"].update(counts)
        rows = scrubbed

    # 6. decontamination against eval sets
    eval_paths = get(cfg, "decontam.eval_files", []) or []
    if steps.get("decontam", True) and eval_paths:
        deco = Decontaminator(load_eval_texts(eval_paths), n=get(cfg, "decontam.ngram", 8))
        kept = [r for r in rows if not deco.is_contaminated(_row_text(r))]
        stats["dropped"]["contaminated"] += len(rows) - len(kept)
        rows = kept

    stats["output"] = len(rows)
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.profile, args.overrides)

    rows: list[dict] = []
    for inp in cfg["inputs"]:
        path = resolve_path(inp["path"] if isinstance(inp, dict) else inp)
        loaded = read_jsonl(path)
        if isinstance(inp, dict) and inp.get("license"):
            loaded = [{**r, "license": r.get("license", inp["license"])} for r in loaded]
        rows.extend(loaded)
    if args.smoke:
        rows = rows[:100]

    clean, stats = run_pipeline(rows, cfg)
    out_path = Path(cfg["output"])
    write_jsonl(out_path, clean)
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input={stats['input']} output={stats['output']} -> {out_path}")
    for reason, n in sorted(stats["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  dropped {reason:<28} {n}")
    for label, n in sorted(stats["pii"].items()):
        print(f"  pii     {label:<28} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
