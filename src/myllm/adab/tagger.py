"""LLM tagger for the *semantic* devices (تشبیه، استعاره، کنایه، مجاز، ایهام) + جناس — data, parsing, scoring.

    python -m myllm.adab.tagger build                                  # -> data/processed/adab_tagger/{train,dev,test}.jsonl
    python -m myllm.train.sft --config configs/adab_tagger.yaml --smoke   # then without --smoke on the 4060
    python -m myllm.adab.tagger eval --split test --model Qwen/Qwen3-4B --adapter out/adab_tagger
    python -m myllm.adab.tagger eval --split test --rules                # rule-based baseline (جناس، تشبیه با ادات)

Gold labels: Ganjoor's LiteraryTags with Origin=AcademicCorpus, i.e. the MIT-licensed «A Corpus of Persian
Literary Text» (Raji, Alikhani, de Melo, Stone 2023; doi:10.5281/zenodo.7923612) for Hafez's ghazals.
Splits are by *poem* (hash of the poem url), so no couplet of a test poem is seen in training.
The corpus does not tag every device, so precision against it is a lower bound; only couplets that carry at
least one tag are used (an untagged couplet is not evidence of "no device").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from myllm.aruz.phonology import MARKS

DEVICES = {1: "تشبیه", 2: "استعاره", 3: "کنایه", 4: "مجاز", 5: "ایهام", 6: "جناس"}
SYSTEM = ("تو استاد بلاغت فارسی هستی. آرایه‌های بیت را فقط با شاهد از خود بیت بنویس؛ "
          "اگر مطمئن نیستی، آن آرایه را ننویس.")
PROMPT = ("آرایه‌های این بیت را بنویس (تشبیه، استعاره، کنایه، مجاز، ایهام، جناس). "
          "هر آرایه در یک سطر، به شکل «نام آرایه: شاهد».\n{m1}\n{m2}")
SOURCE = "persian-literary-corpus-1.0.0 (MIT) via ganjoor-data"
OUT = Path("data/processed/adab_tagger")
_LINE = re.compile(r"^\s*[-•*]?\s*(" + "|".join(DEVICES.values()) + r")\s*[:：]\s*(.*)$")
_QUOTED = re.compile(r"«([^»]*)»")


def split_of(url: str, dev: int = 10, test: int = 10) -> str:
    h = int(hashlib.sha1(url.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < test else ("dev" if h < test + dev else "train")


def _words(text: str) -> list[str]:
    return text.split()


def _span(misras: dict[int, list[str]], part: dict) -> str | None:
    m, a = part.get("Misra"), part.get("WordStart")
    if m not in (1, 2) or not a:
        return None
    b = part.get("WordEnd", a)
    return " ".join(misras[m][a - 1 : b]) or None


def describe(tag: dict, misras: dict[int, list[str]], concepts: dict[int, str]) -> str | None:
    """One gold tag -> «نام: شاهد» line, the same shape the model is trained to write."""
    name = DEVICES.get(tag["DeviceId"])
    if name is None:
        return None
    slots: dict[int, list[str]] = defaultdict(list)
    for p in tag.get("Parts", []):
        val = _span(misras, p) or p.get("Text") or concepts.get(p.get("ConceptId", -1))
        if val:
            slots[p["SlotId"]].append(val.strip())

    def first(slot: int) -> str | None:
        return slots[slot][0] if slots.get(slot) else None

    if name == "تشبیه" and first(1) and first(2):
        s = f"تشبیه: «{first(1)}» به «{first(2)}»"
        return s + (f" (ادات «{first(4)}»)" if first(4) else "") + (f"؛ وجه شبه: {first(3)}" if first(3) else "")
    if name == "استعاره" and first(5):
        return f"استعاره: «{first(5)}»" + (f" برای «{first(6)}»" if first(6) else "")
    if name == "کنایه" and first(7):
        return f"کنایه: «{first(7)}»" + (f" یعنی «{first(8)}»" if first(8) else "")
    if name == "مجاز" and first(9):
        return f"مجاز: «{first(9)}»" + (f" به معنای «{first(10)}»" if first(10) else "")
    if name == "ایهام" and first(11):
        means = slots.get(12, [])
        return f"ایهام: «{first(11)}»" + (" در معناهای " + "، ".join(f"«{x}»" for x in means) if means else "")
    if name == "جناس" and first(13) and first(14):
        return f"جناس: «{first(13)}» و «{first(14)}»"
    return None


def examples(root: str | Path, concepts: dict[int, str]) -> list[dict]:
    rows = []
    for f in sorted(Path(root).rglob("*.json")):
        poem = json.loads(f.read_text(encoding="utf-8"))
        tags = [t for t in poem.get("LiteraryTags") or [] if t.get("Origin") == "AcademicCorpus"]
        if not tags:
            continue
        couplets: dict[int, dict[str, str]] = defaultdict(dict)
        for v in poem.get("Verses", []):
            if v.get("Position") in ("Right", "Left"):
                couplets[v["CoupletIndex"]][v["Position"]] = v["Text"]
        by_couplet: dict[int, list[str]] = defaultdict(list)
        for t in tags:
            c = couplets.get(t["CoupletIndex"], {})
            if len(c) < 2:
                continue
            line = describe(t, {1: _words(c["Right"]), 2: _words(c["Left"])}, concepts)
            if line and line not in by_couplet[t["CoupletIndex"]]:
                by_couplet[t["CoupletIndex"]].append(line)
        url = poem.get("FullUrl", str(f))
        for ci, lines in sorted(by_couplet.items()):
            m1, m2 = couplets[ci]["Right"], couplets[ci]["Left"]
            rows.append({"messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": PROMPT.format(m1=m1, m2=m2)},
                                      {"role": "assistant", "content": "\n".join(lines)}],
                         "id": f"{url}#{ci}", "source": SOURCE, "license": "MIT (annotations)",
                         "split": split_of(url), "verse": [m1, m2]})
    return rows


def build(root: str = "data/raw/ganjoor/hafez", out: Path = OUT,
          concepts_path: str = "data/raw/ganjoor_meta/literary-concepts.json") -> dict[str, int]:
    concepts = {}
    if Path(concepts_path).exists():
        concepts = {c["Id"]: c["Name"] for c in json.loads(Path(concepts_path).read_text(encoding="utf-8"))}
    rows = examples(root, concepts)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    handles = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    try:
        for r in rows:
            handles[r["split"]].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[r["split"]] += 1
    finally:
        for h in handles.values():
            h.close()
    return dict(counts)


# --- parsing and scoring ------------------------------------------------------------------------------
def _norm_tokens(text: str) -> set[str]:
    clean = "".join(ch for ch in text if ch not in MARKS).replace("‌", "").replace("ي", "ی").replace("ك", "ک")
    return {t.strip("«»،؛.!?:()") for t in clean.split()} - {""}


def parse(text: str) -> list[tuple[str, set[str]]]:
    """Model output -> [(device, evidence tokens)] (evidence = words inside «…», else the rest of the line)."""
    out = []
    for line in (text or "").splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        quoted = _QUOTED.findall(m.group(2))
        out.append((m.group(1), _norm_tokens(" ".join(quoted[:2]) if quoted else m.group(2))))
    return out


def _match(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.5 or a <= b or b <= a


def score(gold: list[str], pred: list[str]) -> dict:
    """Per-couplet gold / predicted texts -> device-level and evidence-level precision/recall/F1."""
    dev = {d: Counter() for d in DEVICES.values()}
    ev = Counter()
    for g_text, p_text in zip(gold, pred):
        g, p = parse(g_text), parse(p_text)
        gs, ps = {d for d, _ in g}, {d for d, _ in p}
        for d in DEVICES.values():
            dev[d]["tp"] += d in gs and d in ps
            dev[d]["fp"] += d in ps and d not in gs
            dev[d]["fn"] += d in gs and d not in ps
        used = set()
        for d, toks in p:
            hit = next((k for k, (gd, gt) in enumerate(g) if k not in used and gd == d and _match(toks, gt)), None)
            if hit is None:
                ev["fp"] += 1
            else:
                ev["tp"] += 1
                used.add(hit)
        ev["fn"] += len(g) - len(used)

    def prf(c: Counter) -> dict:
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
                "support": c["tp"] + c["fn"]}

    per = {d: prf(c) for d, c in dev.items()}
    micro = prf(sum(dev.values(), Counter()))
    macro_f1 = round(sum(v["f1"] for v in per.values() if v["support"]) / max(1, sum(1 for v in per.values() if v["support"])), 3)
    return {"device_micro": micro, "device_macro_f1": macro_f1, "per_device": per, "evidence": prf(ev), "n": len(gold)}


def rules_prediction(verse: list[str]) -> str:
    """Baseline from the rule engine: جناس (اختلافی/افزایشی/حرکتی) and تشبیه with an explicit ادات."""
    from myllm.adab.devices import find_devices

    lines = []
    for d in find_devices(verse):
        if d.kind.startswith("جناس ناقص"):
            lines.append(f"جناس: «{d.words[0].plain}» و «{d.words[1].plain}»")
        elif d.kind == "تشبیه":
            lines.append("تشبیه: " + " ".join(f"«{w.plain}»" for w in d.words))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default="data/raw/ganjoor/hafez")
    e = sub.add_parser("eval")
    e.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    e.add_argument("--rules", action="store_true", help="score the rule-based baseline instead of a model")
    e.add_argument("--model", default="tiny://qwen3")
    e.add_argument("--adapter")
    e.add_argument("--base-url", help="OpenAI-compatible server (vLLM/Ollama) instead of local transformers")
    e.add_argument("--limit", type=int)
    e.add_argument("--out", default="out/adab/tagger_eval.json")
    a = p.parse_args(argv)
    if a.cmd == "build":
        print(build(a.root), "->", OUT)
        return 0
    rows = [json.loads(x) for x in (OUT / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[: a.limit] if a.limit else rows
    gold = [r["messages"][-1]["content"] for r in rows]
    if a.rules:
        pred = [rules_prediction(r["verse"]) for r in rows]
    else:
        from myllm.clients import make_client

        block = ({"type": "openai", "base_url": a.base_url, "model": a.model} if a.base_url else
                 {"type": "hf", "model": a.model, "adapter": a.adapter, "max_new_tokens": 160,
                  "chat_template_kwargs": {"enable_thinking": False}})
        client = make_client(block)
        pred = [client.chat(r["messages"][:-1], n=1, temperature=0.0, max_tokens=160)[0] for r in rows]
    result = score(gold, pred)
    result["system"] = "rules" if a.rules else (a.adapter or a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("system", "n", "device_micro", "device_macro_f1", "evidence")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
