"""«ادب‌یار» B2B REST API — prosody, scansion diagnosis, literary devices, question banks and exams.

    python -m myllm.adab.api new-key --name "ناشر نمونه" --plan pro      # prints a key once + the line to store
    ADAB_API_KEYS=configs/api_keys.json python -m myllm.adab.api serve --port 8090
    curl -H "X-API-Key: <key>" -H "Content-Type: application/json" \
         -d '{"lines": ["بشنو این نی چون شکایت می‌کند", "از جدایی‌ها حکایت می‌کند"]}' localhost:8090/v1/scan

For publishers, test-prep schools and ed-tech apps. What the server guarantees:
  * auth      — every /v1 route needs ``X-API-Key``; the keys file holds only SHA-256 hashes, so a leaked
                file leaks no key (see configs/api_keys.example.json)
  * limits    — token bucket per key (``rate_per_minute``) → 429; monthly quota per key → 402
  * metering  — calls counted per key / month / endpoint (optional JSON file) → GET /v1/usage, invoices
  * privacy   — request bodies are never stored or logged
"""

# NOTE: no `from __future__ import annotations` — FastAPI resolves endpoint annotations at runtime.
import argparse
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from myllm.serve.gateway import TokenBucket

PLANS = {  # defaults; a key entry may override rate_per_minute / monthly_quota
    "trial": {"rate_per_minute": 20, "monthly_quota": 2_000},
    "school": {"rate_per_minute": 60, "monthly_quota": 30_000},
    "pro": {"rate_per_minute": 300, "monthly_quota": 500_000},
}


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ApiKey:
    name: str
    sha256: str
    plan: str = "trial"
    rate_per_minute: int = 0
    monthly_quota: int = 0

    def __post_init__(self):
        base = PLANS.get(self.plan, PLANS["trial"])
        self.rate_per_minute = self.rate_per_minute or base["rate_per_minute"]
        self.monthly_quota = self.monthly_quota or base["monthly_quota"]


class KeyStore:
    def __init__(self, keys: list[ApiKey]):
        self.keys = keys

    @classmethod
    def load(cls, path: str | Path) -> "KeyStore":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([ApiKey(**k) for k in raw.get("keys", [])])

    def find(self, presented: str | None) -> ApiKey | None:
        if not presented:
            return None
        digest = _sha(presented)
        for k in self.keys:  # constant-time compare per entry
            if secrets.compare_digest(k.sha256, digest):
                return k
        return None


class Meter:
    """usage[name][YYYY-MM][endpoint] = calls; persisted atomically when a path is given."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.usage: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}

    @staticmethod
    def month(now: float | None = None) -> str:
        return time.strftime("%Y-%m", time.gmtime(now))

    def total(self, name: str, month: str | None = None) -> int:
        return sum(self.usage.get(name, {}).get(month or self.month(), {}).values())

    def add(self, name: str, endpoint: str) -> None:
        with self.lock:
            m = self.usage.setdefault(name, {}).setdefault(self.month(), {})
            m[endpoint] = m.get(endpoint, 0) + 1
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.usage, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.path)


def _scan_json(r) -> dict:
    return {"text": r.text, "ok": r.ok, "meter": r.meter.name if r.meter else None,
            "feet": r.meter.feet_label() if r.meter else None, "syllables": r.syllables, "weights": r.weights,
            "ikhtiyarat": r.ikhtiyarat, "confidence": round(r.confidence, 3)}


def create_app(keys: KeyStore, scanner=None, banks: list | None = None, meter: Meter | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    from myllm.adab.devices import find_devices
    from myllm.adab.exam import build_exam
    from myllm.aruz.diagnose import diagnose
    from myllm.aruz.rhyme import find_rhyme

    if scanner is None:
        from myllm.aruz.engine import build_scanner

        scanner = build_scanner()
    meter = meter or Meter()
    banks = banks or []
    buckets = {k.name: TokenBucket(k.rate_per_minute) for k in keys.keys}
    lock = threading.Lock()  # the scanner keeps per-call state; serialise scans
    app = FastAPI(title="Adab-yar API", version="1.0")

    def client(endpoint: str):
        def dep(x_api_key: str | None = Header(default=None)) -> ApiKey:
            k = keys.find(x_api_key)
            if k is None:
                raise HTTPException(401, "missing or unknown X-API-Key")
            if not buckets.setdefault(k.name, TokenBucket(k.rate_per_minute)).allow(k.name):
                raise HTTPException(429, "rate limit exceeded")
            if meter.total(k.name) >= k.monthly_quota:
                raise HTTPException(402, "monthly quota used up")
            meter.add(k.name, endpoint)
            return k
        return dep

    class Lines(BaseModel):
        lines: list[str] = Field(min_length=1, max_length=12)

    class Check(BaseModel):
        hemistich: str = Field(min_length=1, max_length=300)
        signs: str = Field(min_length=1, max_length=200)

    class Exam(BaseModel):
        mix: str = "meter:5,ikhtiyar:2,history:8"
        versions: int = Field(default=1, ge=1, le=4)
        title: str = "آزمون علوم و فنون ادبی"
        school: str = ""
        minutes: int | None = 40
        seed: int = 42

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.post("/v1/scan")
    def scan(body: Lines, k: ApiKey = Depends(client("scan"))):
        lines = [x.strip() for x in body.lines if x.strip()]
        if any(len(x) > 300 for x in lines):
            raise HTTPException(422, "a hemistich longer than 300 characters")
        with lock:
            if len(lines) == 1:
                results = [scanner.scan(lines[0])]
            elif len(lines) == 2:
                results = list(scanner.scan_couplet(lines[0], lines[1]))
            else:
                results = scanner.scan_poem(lines)
        rhyme = find_rhyme(lines if len(lines) <= 2 else [x for i, x in enumerate(lines) if i < 2 or i % 2 == 1])
        return {"meter": results[0].meter.name if results[0].ok else None,
                "feet": results[0].meter.feet_label() if results[0].ok else None,
                "confidence": round(results[0].confidence, 3), "hemistichs": [_scan_json(r) for r in results],
                "rhyme": {"radif": rhyme.radif, "qafiye": rhyme.qafiye_words, "letters": rhyme.letters}
                if len(lines) >= 2 else None}

    @app.post("/v1/check")
    def check(body: Check, k: ApiKey = Depends(client("check"))):
        with lock:
            d = diagnose(scanner, body.hemistich, body.signs)
        return {"expected": d.expected, "student": d.student, "score": round(d.score, 3), "found": bool(d.expected),
                "mistakes": [{"position": m.position, "syllable": m.syllable, "expected": m.expected, "got": m.got,
                              "rule": m.rule, "reason": m.reason} for m in d.mistakes]}

    @app.post("/v1/devices")
    def devices(body: Lines, k: ApiKey = Depends(client("devices"))):
        if len(body.lines) > 2:
            raise HTTPException(422, "send one couplet (two hemistichs)")
        return {"devices": [{"kind": d.kind, "confidence": d.confidence, "explanation": d.explanation,
                             "words": [{"misra": w.misra, "index": w.index, "text": w.raw} for w in d.words]}
                            for d in find_devices(body.lines)]}

    @app.get("/v1/questions")
    def questions(kind: str = Query(default=""), n: int = Query(default=10, ge=1, le=100), seed: int = 0,
                  k: ApiKey = Depends(client("questions"))):
        import random
        from dataclasses import asdict

        pool = [q for q in banks if not kind or q.kind == kind]
        rng = random.Random(seed)
        return {"questions": [asdict(q) for q in rng.sample(pool, min(n, len(pool)))]}

    @app.post("/v1/exam", response_class=HTMLResponse)
    def exam(body: Exam, k: ApiKey = Depends(client("exam"))):
        try:
            doc, _ = build_exam(banks, body.mix, body.versions, body.seed, body.title, body.school, body.minutes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return HTMLResponse(doc)

    @app.get("/v1/usage")
    def usage(k: ApiKey = Depends(client("usage"))):
        return {"name": k.name, "plan": k.plan, "month": Meter.month(), "used": meter.total(k.name),
                "monthly_quota": k.monthly_quota, "by_endpoint": meter.usage.get(k.name, {}).get(Meter.month(), {})}

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    nk = sub.add_parser("new-key")
    nk.add_argument("--name", required=True)
    nk.add_argument("--plan", choices=sorted(PLANS), default="trial")
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8090)
    sv.add_argument("--keys", default=os.environ.get("ADAB_API_KEYS", "configs/api_keys.json"))
    sv.add_argument("--usage", default="out/api/usage.json")
    sv.add_argument("--banks", default="data/processed/aruz_exercises.jsonl,data/processed/history_bank.jsonl")
    a = p.parse_args(argv)
    if a.cmd == "new-key":
        key = "adab_" + secrets.token_urlsafe(24)
        print(f"API key (shown once, give it to the customer): {key}")
        print("add to the keys file:", json.dumps({"name": a.name, "sha256": _sha(key), "plan": a.plan}, ensure_ascii=False))
        return 0
    if not Path(a.keys).exists():
        raise SystemExit(f"no keys file at {a.keys}; create one with `new-key` (see configs/api_keys.example.json)")
    import uvicorn

    from myllm.aruz.exercises import load_bank

    banks = [q for path in a.banks.split(",") for q in load_bank(path)]
    app = create_app(KeyStore.load(a.keys), banks=banks, meter=Meter(a.usage))
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
