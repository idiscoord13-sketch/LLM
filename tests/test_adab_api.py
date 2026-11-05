"""B2B API (myllm.adab.api): auth, limits, metering and the main endpoints."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from myllm.adab.api import ApiKey, KeyStore, Meter, _sha, create_app  # noqa: E402
from myllm.aruz.scan import Scanner  # noqa: E402


@pytest.fixture(scope="module")
def scanner():
    return Scanner()


def _client(scanner, tmp_path, **key):
    from myllm.adab.history import build

    keys = KeyStore([ApiKey(name="pub", sha256=_sha("secret-1"), plan="pro", **key)])
    meter = Meter(tmp_path / "usage.json")
    app = create_app(keys, scanner=scanner, banks=build(60, seed=1), meter=meter)
    return TestClient(app), meter


H = {"X-API-Key": "secret-1"}


def test_auth_and_scan(scanner, tmp_path):
    c, _ = _client(scanner, tmp_path)
    assert c.get("/healthz").json() == {"ok": True}
    body = {"lines": ["بشنو این نی چون شکایت می‌کند", "از جدایی‌ها حکایت می‌کند"]}
    assert c.post("/v1/scan", json=body).status_code == 401
    assert c.post("/v1/scan", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    r = c.post("/v1/scan", json=body, headers=H)
    assert r.status_code == 200
    out = r.json()
    assert "رمل" in out["meter"] and len(out["hemistichs"]) == 2
    assert out["rhyme"]["radif"] == "می‌کند" or "کند" in out["rhyme"]["radif"]


def test_check_devices_questions_exam_usage(scanner, tmp_path):
    c, meter = _client(scanner, tmp_path)
    r = c.post("/v1/check", json={"hemistich": "بشنو این نی چون شکایت می‌کند", "signs": "– U – U – U – – – U –"},
               headers=H).json()
    assert r["found"] and r["mistakes"] and all(m["rule"] for m in r["mistakes"])
    d = c.post("/v1/devices", json={"lines": ["هر آن که کنج قناعت به گنج دنیا داد", "فروخت یوسف مصری به کمترین ثمنی"]},
               headers=H).json()["devices"]
    assert any(x["kind"] == "جناس ناقص اختلافی" for x in d)
    qs = c.get("/v1/questions?kind=century&n=3", headers=H).json()["questions"]
    assert len(qs) == 3 and all(q["kind"] == "century" for q in qs)
    html = c.post("/v1/exam", json={"mix": "history:5", "versions": 2}, headers=H)
    assert html.status_code == 200 and "پاسخ‌نامه" in html.text
    assert c.post("/v1/exam", json={"mix": "meter:5"}, headers=H).status_code == 422
    u = c.get("/v1/usage", headers=H).json()
    assert u["by_endpoint"]["check"] == 1 and u["used"] == 6  # 5 calls before + this one (a 422 still counts)
    assert '"secret-1"' not in (tmp_path / "usage.json").read_text(encoding="utf-8")


def test_rate_limit_and_quota(scanner, tmp_path):
    c, _ = _client(scanner, tmp_path, rate_per_minute=2)
    codes = [c.get("/v1/usage", headers=H).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    c2, _ = _client(scanner, tmp_path / "q", monthly_quota=1)
    assert c2.get("/v1/usage", headers=H).status_code == 200
    assert c2.get("/v1/usage", headers=H).status_code == 402
