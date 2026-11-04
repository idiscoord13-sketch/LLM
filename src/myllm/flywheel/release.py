"""Model registry, canary rollout and rollback (flywheel step 8).

    python -m myllm.flywheel.release register --version v2 --upstream http://gpu:8000/v1 --model v2 --results out/eval/results.json
    python -m myllm.flywheel.release canary   --version v2 --percent 10
    python -m myllm.flywheel.release promote  --version v2
    python -m myllm.flywheel.release rollback
    python -m myllm.flywheel.release status

The gateway re-reads ``registry.json`` on every request (cheap), so a rollback takes
effect immediately without restarting anything. Users are assigned to canary by a
stable hash of their id, so one user does not flip between versions mid-conversation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

DEFAULT_REGISTRY = Path("data/flywheel/registry.json")


class Registry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "versions": {}, "stable": None, "canary": None, "history": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic swap: the gateway never reads a half-written file

    def _event(self, action: str, **kw) -> None:
        self.data["history"].append({"ts": time.time(), "action": action, **kw})

    def register(self, version: str, upstream: str, model: str, results: dict | None = None,
                 api_key_env: str | None = None) -> None:
        if version in self.data["versions"]:
            raise ValueError(f"version {version} already registered")
        self.data["versions"][version] = {"upstream": upstream, "model": model, "results": results or {},
                                          "api_key_env": api_key_env, "status": "candidate", "created": time.time()}
        if self.data["stable"] is None:
            self.data["stable"] = version
            self.data["versions"][version]["status"] = "stable"
        self._event("register", version=version)
        self.save()

    def start_canary(self, version: str, percent: float) -> None:
        if version not in self.data["versions"]:
            raise KeyError(version)
        if not 0 < percent <= 50:
            raise ValueError("canary percent must be in (0, 50]")
        self.data["canary"] = {"version": version, "percent": percent}
        self.data["versions"][version]["status"] = "canary"
        self._event("canary", version=version, percent=percent)
        self.save()

    def promote(self, version: str) -> None:
        old = self.data["stable"]
        if old:
            self.data["versions"][old]["status"] = "retired"
        self.data["stable"] = version
        self.data["versions"][version]["status"] = "stable"
        if self.data["canary"] and self.data["canary"]["version"] == version:
            self.data["canary"] = None
        self._event("promote", version=version, previous=old)
        self.save()

    def rollback(self) -> str | None:
        """Abort an active canary; if none, restore the previous stable version."""
        if self.data["canary"]:
            v = self.data["canary"]["version"]
            self.data["versions"][v]["status"] = "rolled_back"
            self.data["canary"] = None
            self._event("rollback_canary", version=v)
            self.save()
            return self.data["stable"]
        promotions = [h for h in self.data["history"] if h["action"] == "promote" and h.get("previous")]
        if not promotions:
            return self.data["stable"]
        last = promotions[-1]
        bad, prev = last["version"], last["previous"]
        self.data["versions"][bad]["status"] = "rolled_back"
        self.data["versions"][prev]["status"] = "stable"
        self.data["stable"] = prev
        self._event("rollback_stable", version=bad, restored=prev)
        self.save()
        return prev

    def route(self, user_key: str) -> str | None:
        """Stable assignment: hash(user) % 10000 < percent*100 -> canary."""
        canary = self.data.get("canary")
        if canary:
            bucket = int(hashlib.sha256(user_key.encode("utf-8")).hexdigest(), 16) % 10000
            if bucket < canary["percent"] * 100:
                return canary["version"]
        return self.data.get("stable")

    def get(self, version: str) -> dict:
        return self.data["versions"][version]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="model registry / canary / rollback")
    p.add_argument("action", choices=["register", "canary", "promote", "rollback", "status"])
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--version")
    p.add_argument("--upstream")
    p.add_argument("--model")
    p.add_argument("--api-key-env")
    p.add_argument("--results")
    p.add_argument("--percent", type=float, default=10.0)
    a = p.parse_args(argv)
    reg = Registry(a.registry)
    if a.action == "register":
        results = json.loads(Path(a.results).read_text(encoding="utf-8")) if a.results else None
        reg.register(a.version, a.upstream, a.model or a.version, results, a.api_key_env)
    elif a.action == "canary":
        reg.start_canary(a.version, a.percent)
    elif a.action == "promote":
        reg.promote(a.version)
    elif a.action == "rollback":
        print("stable is now", reg.rollback())
    print(json.dumps({k: reg.data[k] for k in ("stable", "canary")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
