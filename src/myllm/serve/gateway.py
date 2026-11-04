"""OpenAI-compatible gateway in front of vLLM / Ollama / llama.cpp (roadmap §9.4, Phase 4-5).

    GATEWAY_API_KEYS=... python -m myllm.serve.gateway --config configs/gateway.yaml
    # then point Open WebUI (or any OpenAI client) at http://localhost:8080/v1

What it adds on top of the model server:
  * auth           — Bearer API key required on every route (never expose a bare model server)
  * rate limiting  — token bucket per API key (requests/minute)
  * routing        — stable + canary versions from the registry (myllm.flywheel.release)
  * consent logging— only if the request carries consent (header ``X-Consent: true`` or
                     ``metadata.consent``); PII scrubbed before anything is written
  * feedback       — POST /v1/feedback {conversation_id, thumb, edit, regenerate}
  * test-time compute (optional) — ``"tts": {"n": 4}`` samples n answers, picks the best
  * GDPR erasure   — POST /v1/forget {user_id}

Streaming is passed through as a single (non-streamed) response in this version.
"""

# NOTE: no `from __future__ import annotations` here — FastAPI must resolve the
# `Request`/`Header` annotations of the endpoints defined inside `create_app`.
import argparse
import os
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from myllm.config import get, load_config
from myllm.flywheel.release import Registry
from myllm.flywheel.store import FlywheelStore
from myllm.serve.tts import select_best


class TokenBucket:
    def __init__(self, rate_per_minute: float, burst: int | None = None):
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst or max(1, int(rate_per_minute)))
        self.state: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            tokens, last = self.state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self.state[key] = (tokens, now)
                return False
            self.state[key] = (tokens - 1.0, now)
            return True


def default_upstream_call(version_cfg: dict, body: dict, timeout: float = 300.0) -> dict:
    import httpx

    key = os.environ.get(version_cfg.get("api_key_env") or "UPSTREAM_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = httpx.post(version_cfg["upstream"].rstrip("/") + "/chat/completions", json=body, headers=headers,
                      timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def create_app(cfg: dict, upstream_call: "Callable[[dict, dict], dict] | None" = None):
    from fastapi import Depends, FastAPI, Header, HTTPException, Request

    keys = {k.strip() for k in (os.environ.get(get(cfg, "auth.keys_env", "GATEWAY_API_KEYS"), "")).split(",") if k.strip()}
    keys |= set(get(cfg, "auth.keys", []) or [])
    if not keys:
        raise RuntimeError("refusing to start without API keys (set GATEWAY_API_KEYS)")
    bucket = TokenBucket(float(get(cfg, "rate_limit.requests_per_minute", 60)), get(cfg, "rate_limit.burst"))
    store = FlywheelStore(get(cfg, "flywheel.store_dir", "data/flywheel"), salt=os.environ.get("FLYWHEEL_SALT", ""))
    registry_path = Path(get(cfg, "registry", "data/flywheel/registry.json"))
    call = upstream_call or default_upstream_call
    max_tts = int(get(cfg, "tts.max_n", 8))

    app = FastAPI(title="myllm gateway")

    def auth(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        if not token or not any(secrets.compare_digest(token, k) for k in keys):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        if not bucket.allow(token):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return token

    def pick_version(user_key: str) -> tuple[str, dict]:
        reg = Registry(registry_path)
        version = reg.route(user_key)
        if version is None:
            static = get(cfg, "static_upstream")
            if not static:
                raise HTTPException(status_code=503, detail="no model registered")
            return "static", static
        return version, reg.get(version)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/models")
    def models(_: str = Depends(auth)):
        reg = Registry(registry_path)
        return {"object": "list", "data": [{"id": "myllm", "object": "model", "stable": reg.data.get("stable")}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, token: str = Depends(auth), x_consent: str = Header(default=""),
                   x_user_id: str = Header(default="")):
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")
        meta = body.pop("metadata", None) or {}
        tts = body.pop("tts", None)
        user_key = x_user_id or body.get("user") or token
        version, vcfg = pick_version(user_key)
        upstream_body = {**body, "model": vcfg.get("model", body.get("model")), "stream": False}

        info = None
        if tts and int(tts.get("n", 1)) > 1:
            n = min(max_tts, int(tts["n"]))
            upstream_body["n"] = n
            upstream_body.setdefault("temperature", 0.8)
            data = call(vcfg, upstream_body)
            cands = [c["message"]["content"] or "" for c in data["choices"]]
            best, info = select_best(messages[-1].get("content", ""), cands, tts.get("kind"))
            data["choices"] = [{**data["choices"][best], "index": 0}]
        else:
            data = call(vcfg, upstream_body)

        conv_id = body.get("conversation_id") or f"conv-{uuid.uuid4().hex[:12]}"
        consent = x_consent.lower() == "true" or bool(meta.get("consent"))
        answer = data["choices"][0]["message"]
        logged = store.log_conversation(conv_id, [*messages, {"role": "assistant", "content": answer.get("content") or ""}],
                                        user_id=user_key if user_key != token else None, consent=consent,
                                        model_version=version)
        data["id"] = conv_id
        data["model"] = f"myllm:{version}"
        data["myllm"] = {"version": version, "logged": logged, **({"tts": info} if info else {})}
        return data

    @app.post("/v1/feedback")
    async def feedback(request: Request, _: str = Depends(auth), x_user_id: str = Header(default="")):
        fb = await request.json()
        if not fb.get("conversation_id"):
            raise HTTPException(status_code=400, detail="conversation_id required")
        store.log_feedback(fb["conversation_id"], thumb=fb.get("thumb"), edit=fb.get("edit"),
                           regenerate=bool(fb.get("regenerate")), user_id=x_user_id or None)
        return {"ok": True}

    @app.post("/v1/forget")
    async def forget(request: Request, _: str = Depends(auth)):
        body = await request.json()
        if not body.get("user_id"):
            raise HTTPException(status_code=400, detail="user_id required")
        return {"removed": store.forget_user(body["user_id"])}

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="myllm OpenAI-compatible gateway")
    p.add_argument("--config", default="configs/gateway.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(load_config(a.config)), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
