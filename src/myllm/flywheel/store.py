"""Consent-aware conversation + feedback store (flywheel steps 1-2, roadmap §7.3).

Layout (JSONL, append-only, one directory per deployment)::

    data/flywheel/conversations.jsonl   # only consented conversations, PII already scrubbed
    data/flywheel/feedback.jsonl        # 👍/👎, regenerate, user edits
    data/flywheel/deletions.jsonl       # right-to-erasure requests (user ids)

Design choices:
  * no consent -> nothing is written (not even "anonymised")
  * PII is scrubbed *before* the first write; raw text never touches disk here
  * ``forget_user`` rewrites the files without that user's rows (GDPR-style erasure)
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from myllm.data.pii import scrub_pii
from myllm.utils import append_jsonl, iter_jsonl, write_jsonl


def pseudonymize(user_id: str | None, salt: str = "") -> str | None:
    if not user_id:
        return None
    return hashlib.sha256((salt + user_id).encode("utf-8")).hexdigest()[:16]


class FlywheelStore:
    def __init__(self, root: str | Path, salt: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.salt = salt
        self._lock = threading.Lock()

    @property
    def conversations_path(self) -> Path:
        return self.root / "conversations.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self.root / "feedback.jsonl"

    def log_conversation(self, conversation_id: str, messages: list[dict], *, user_id: str | None,
                         consent: bool, model_version: str | None, meta: dict | None = None) -> bool:
        if not consent:
            return False
        clean = [{"role": m["role"], "content": scrub_pii(m.get("content") or "")[0]} for m in messages]
        row = {"conversation_id": conversation_id, "user_id": pseudonymize(user_id, self.salt), "consent": True,
               "model_version": model_version, "messages": clean, "ts": time.time(), **(meta or {})}
        with self._lock:
            append_jsonl(self.conversations_path, row)
        return True

    def log_feedback(self, conversation_id: str, *, thumb: str | None = None, edit: str | None = None,
                     regenerate: bool = False, user_id: str | None = None) -> None:
        row = {"conversation_id": conversation_id, "user_id": pseudonymize(user_id, self.salt), "ts": time.time()}
        if thumb in ("up", "down"):
            row["thumb"] = thumb
        if edit:
            row["edit"] = scrub_pii(edit)[0]
        if regenerate:
            row["regenerate"] = True
        with self._lock:
            append_jsonl(self.feedback_path, row)

    def load(self) -> list[dict]:
        """Conversations joined with their (merged) feedback."""
        if not self.conversations_path.exists():
            return []
        feedback: dict[str, dict] = {}
        if self.feedback_path.exists():
            for fb in iter_jsonl(self.feedback_path):
                merged = feedback.setdefault(fb["conversation_id"], {})
                merged.update({k: v for k, v in fb.items() if k in ("thumb", "edit", "regenerate")})
        rows = []
        for conv in iter_jsonl(self.conversations_path):
            rows.append({**conv, "feedback": feedback.get(conv["conversation_id"], {})})
        return rows

    def forget_user(self, user_id: str) -> int:
        """Erase every row of a user. Returns the number of rows removed."""
        pid = pseudonymize(user_id, self.salt)
        removed = 0
        with self._lock:
            for path in (self.conversations_path, self.feedback_path):
                if not path.exists():
                    continue
                rows = list(iter_jsonl(path))
                keep = [r for r in rows if r.get("user_id") != pid]
                removed += len(rows) - len(keep)
                write_jsonl(path, keep)
            append_jsonl(self.root / "deletions.jsonl", {"user_id": pid, "ts": time.time()})
        return removed

    def import_rows(self, rows: list[dict]) -> int:
        """Load pre-collected logs (e.g. data/samples/flywheel_logs.jsonl) through the same consent/PII gate."""
        n = 0
        for r in rows:
            if self.log_conversation(r["conversation_id"], r["messages"], user_id=r.get("user_id"),
                                     consent=bool(r.get("consent")), model_version=r.get("model_version")):
                n += 1
                fb = r.get("feedback") or {}
                if fb:
                    self.log_feedback(r["conversation_id"], thumb=fb.get("thumb"), edit=fb.get("edit"),
                                      regenerate=bool(fb.get("regenerate")), user_id=r.get("user_id"))
        return n


def dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)
