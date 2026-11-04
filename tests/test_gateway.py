import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from myllm.flywheel.release import Registry  # noqa: E402
from myllm.flywheel.store import FlywheelStore  # noqa: E402
from myllm.serve.gateway import TokenBucket, create_app  # noqa: E402
from myllm.serve.tts import select_best  # noqa: E402

KEY = "test-key-123"


def fake_upstream(calls):
    def call(vcfg, body):
        calls.append((vcfg["model"], body))
        n = body.get("n", 1)
        contents = [f"answer from {vcfg['model']}"] * n
        if n > 1:
            contents = ["```python\nthis is not python\n```"] + ["```python\nprint(1)\n```"] * (n - 1)
        return {"choices": [{"index": i, "message": {"role": "assistant", "content": c}} for i, c in enumerate(contents)]}
    return call


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEYS", KEY)
    reg = Registry(tmp_path / "registry.json")
    reg.register("v1", "http://stable/v1", "stable-model")
    cfg = {"registry": str(tmp_path / "registry.json"), "flywheel": {"store_dir": str(tmp_path / "fw")},
           "rate_limit": {"requests_per_minute": 600, "burst": 5}}
    calls = []
    client = TestClient(create_app(cfg, upstream_call=fake_upstream(calls)))
    return client, calls, tmp_path, reg


def chat(client, consent=None, user="alice", **extra):
    headers = {"Authorization": f"Bearer {KEY}", "X-User-Id": user}
    if consent is not None:
        headers["X-Consent"] = "true" if consent else "false"
    body = {"messages": [{"role": "user", "content": "سلام، ایمیل من a@b.com است"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_requires_api_key(env):
    client, *_ = env
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert client.get("/health").status_code == 200


def test_refuses_to_start_without_keys(monkeypatch):
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        create_app({})


def test_rate_limit(env):
    client, *_ = env
    codes = [chat(client).status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


def test_token_bucket_refills():
    b = TokenBucket(60, burst=1)
    assert b.allow("k", now=0.0) and not b.allow("k", now=0.1) and b.allow("k", now=1.1)


def test_consent_logging_and_pii(env):
    client, _, tmp, _ = env
    r = chat(client, consent=False)
    assert r.status_code == 200 and r.json()["myllm"]["logged"] is False
    r = chat(client, consent=True)
    assert r.json()["myllm"]["logged"] is True
    store = FlywheelStore(tmp / "fw")
    rows = store.load()
    assert len(rows) == 1 and "a@b.com" not in str(rows) and "[EMAIL]" in str(rows)
    fb = client.post("/v1/feedback", json={"conversation_id": r.json()["id"], "thumb": "up"},
                     headers={"Authorization": f"Bearer {KEY}"})
    assert fb.status_code == 200 and store.load()[0]["feedback"]["thumb"] == "up"


def test_canary_routing(env):
    client, calls, _, reg = env
    reg.register("v2", "http://canary/v1", "canary-model")
    reg.start_canary("v2", 50)
    models = set()
    for i in range(30):
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]},
                        headers={"Authorization": f"Bearer {KEY}", "X-User-Id": f"user-{i}"})
        models.add(r.json()["myllm"]["version"])
        if len(models) == 2:
            break
    assert models == {"v1", "v2"}


def test_test_time_compute_picks_running_code(env):
    client, calls, *_ = env
    r = chat(client, tts={"n": 3, "kind": "code"})
    assert "print(1)" in r.json()["choices"][0]["message"]["content"]
    assert calls[-1][1]["n"] == 3


def test_select_best_math_majority():
    idx, info = select_best("۲۳ ضربدر ۴۷ چند می‌شود؟", ["جواب 1081", "جواب ۱۰۸۱ است", "1080"])
    assert idx in (0, 1) and info["kind"] == "math"
