"""«عروض‌یار» chat bot for Bale or Telegram (same Bot API shape), via long polling.

    ARUZ_BOT_TOKEN=... python -m myllm.aruz.bot --platform bale --lexicon data/processed/aruz_lexicon.json
    ARUZ_BOT_TOKEN=... python -m myllm.aruz.bot --platform telegram

Send one hemistich, or a couplet as two lines (or separated by « / » or «**»), and get back the
metre, the syllable division, feet, اختیارات and rhyme. Nothing is stored; no LLM, no GPU.
Usage counts per day are kept in memory only (for your demo metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

from myllm.aruz.lexicon import load_lexicon
from myllm.aruz.rhyme import find_rhyme, format_rhyme
from myllm.aruz.scan import Scanner, format_result

API_BASE = {"bale": "https://tapi.bale.ai", "telegram": "https://api.telegram.org"}
HELP = (
    "سلام! من «عروض‌یار» هستم.\n"
    "یک مصراع یا یک بیت (دو مصراع در دو خط) بفرست تا وزن، تقطیع هجایی، ارکان، اختیارات شاعری و قافیه‌اش را بگویم.\n"
    "اگر چند بیت از یک شعر (تا ۶ بیت) بفرستی، دقت تشخیص وزن خیلی بیشتر می‌شود.\n\n"
    "مثال:\n"
    "بشنو این نی چون شکایت می‌کند\n"
    "از جدایی‌ها حکایت می‌کند\n\n"
    "🔎 بررسی تقطیع خودت: /check در خط اول، مصراع در خط دوم، نشانه‌هایت در خط سوم (مثل: – U – –)\n"
    "📝 تمرین: /tamrin یک سؤال می‌دهد؛ جواب را با شماره‌ی گزینه (۱ تا ۴) بفرست.\n\n"
    "⚠️ نسخه‌ی آزمایشی است؛ برای بیت‌های عربی، نیمایی و سپید وزن عروضی معنا ندارد یا ممکن است اشتباه کنم."
)
MAX_LINES = 12
_SPLIT = re.compile(r"\s*(?:\n+|\s/\s|\*\*|\s\|\s)\s*")


_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)


class AruzService:
    def __init__(self, lexicon_path: str | None = None, exercises_path: str | None = None,
                 scanner: Scanner | None = None, seed: int = 0, ensemble_path: str | None = None,
                 classifier_path: str | None = None):
        if scanner is None:
            if ensemble_path or classifier_path:
                from myllm.aruz.engine import build_scanner

                scanner = build_scanner(lexicon_path, ensemble_path, classifier_path)
            else:
                scanner = Scanner(load_lexicon(lexicon_path))
        self.scanner = scanner
        self.stats: Counter = Counter()
        from myllm.aruz.exercises import load_bank

        self.bank = [q for q in load_bank(exercises_path) if q.answer is not None] if exercises_path else []
        self.pending: dict = {}  # chat id -> question waiting for an answer
        self.rng = random.Random(seed)

    def handle_text(self, text: str, chat_id=None) -> str:
        text = (text or "").strip()
        if not text or text.startswith("/start") or text.startswith("/help"):
            return HELP
        if text.startswith("/check"):
            return self._check(text[len("/check"):].strip())
        if text.startswith("/tamrin"):
            return self._quiz(chat_id)
        answer = text.translate(_DIGITS).strip()
        if chat_id in self.pending and answer in {"1", "2", "3", "4"}:
            return self._grade(chat_id, int(answer) - 1)
        if text.startswith("/vazn"):
            text = text[len("/vazn"):].strip()
        parts = [p for p in _SPLIT.split(text) if p.strip()]
        if len(parts) > MAX_LINES or len(text) > 2000:
            self.stats["too_long"] += 1
            return f"لطفاً حداکثر {MAX_LINES // 2} بیت ({MAX_LINES} مصراع) بفرست."
        self.stats["requests"] += 1
        if len(parts) > 2:
            return self._poem(parts)
        if len(parts) == 2:
            r1, r2 = self.scanner.scan_couplet(parts[0], parts[1])
            rhyme = find_rhyme(parts)
            body = format_result(r1) + "\n\n" + format_result(r2)
            if rhyme.ok or rhyme.radif:
                body += "\n\n" + format_rhyme(rhyme)
            self.stats["found" if r1.ok else "not_found"] += 1
            return body
        r = self.scanner.scan(parts[0])
        self.stats["found" if r.ok else "not_found"] += 1
        return format_result(r) + "\n\n(برای دقت بیشتر، هر دو مصراع بیت را بفرست.)"


    def _check(self, body: str) -> str:
        from myllm.aruz.diagnose import diagnose, format_diagnosis, parse_signs

        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) < 2 or not parse_signs(lines[-1]):
            return ("برای بررسی تقطیع این‌طور بفرست:\n/check\nبشنو این نی چون شکایت می‌کند\n– U – – – U – – – U –")
        self.stats["check"] += 1
        return format_diagnosis(diagnose(self.scanner, lines[0], lines[-1]))

    def _quiz(self, chat_id) -> str:
        if not self.bank:
            return "بانک تمرین هنوز ساخته نشده (python -m myllm.aruz.exercises)."
        q = self.rng.choice(self.bank)
        self.pending[chat_id] = q
        self.stats["quiz"] += 1
        opts = "\n".join(f"{i + 1}) {o}" for i, o in enumerate(q.options))
        return f"{q.prompt}\n\n" + "\n".join(q.verse) + f"\n\n{opts}\n\nجواب را با شماره‌ی گزینه بفرست."

    def _grade(self, chat_id, choice: int) -> str:
        q = self.pending.pop(chat_id)
        ok = choice == q.answer
        self.stats["quiz_correct" if ok else "quiz_wrong"] += 1
        head = "✅ درست است!" if ok else f"❌ نه؛ جواب درست گزینه‌ی {q.answer + 1} است: {q.options[q.answer]}"
        return head + (f"\n{q.explanation}" if q.explanation else "") + "\n\nسؤال بعدی: /tamrin"

    def _poem(self, parts: list[str]) -> str:
        """Several couplets of one poem: one metre for all (much more reliable than one couplet)."""
        results = self.scanner.scan_poem(parts)
        first = results[0]
        self.stats["poem"] += 1
        if not first.ok:
            return "وزنی پیدا نشد (ممکن است شعر نیمایی یا سپید باشد)."
        level = "بالا" if first.confidence >= 0.7 else "متوسط" if first.confidence >= 0.4 else "پایین"
        lines = [f"وزن کل شعر ({len(parts)} مصراع): {first.meter.feet_label()} — {first.meter.name}", f"اطمینان: {level}", ""]
        lines.append(format_result(results[0]))
        if len(results) > 1:
            lines += ["", format_result(results[1])]
        rhyme = find_rhyme([p for k, p in enumerate(parts) if k < 2 or k % 2 == 1])
        if rhyme.ok or rhyme.radif:
            lines += ["", format_rhyme(rhyme)]
        return "\n".join(lines)


def _call(base: str, token: str, method: str, params: dict, timeout: float = 70) -> dict:
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(f"{base}/bot{token}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def poll(token: str, platform: str = "bale", lexicon: str | None = None, exercises: str | None = None,
         ensemble: str | None = None, classifier: str | None = None) -> None:  # pragma: no cover - network
    base = API_BASE[platform]
    service = AruzService(lexicon, exercises, ensemble_path=ensemble, classifier_path=classifier)
    offset = 0
    print(f"aruz bot running on {platform}…")
    while True:
        try:
            updates = _call(base, token, "getUpdates", {"offset": offset, "timeout": 50})
            for up in updates.get("result", []):
                offset = up["update_id"] + 1
                msg = up.get("message") or {}
                if "text" not in msg:
                    continue
                reply = service.handle_text(msg["text"], chat_id=msg["chat"]["id"])
                _call(base, token, "sendMessage", {"chat_id": msg["chat"]["id"], "text": reply,
                                                   "reply_to_message_id": msg.get("message_id")})
        except Exception as exc:  # network hiccups: keep polling
            print("poll error:", exc)
            time.sleep(3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=sorted(API_BASE), default="bale")
    p.add_argument("--lexicon", default="data/processed/aruz_lexicon.json")
    p.add_argument("--exercises", default="data/processed/aruz_exercises.jsonl")
    p.add_argument("--ensemble", default="data/processed/aruz_ensemble.json")
    p.add_argument("--classifier", default="out/aruz/classifier.pt")
    p.add_argument("--repl", action="store_true", help="local console instead of a bot (no token needed)")
    a = p.parse_args(argv)
    if a.repl:
        service = AruzService(a.lexicon, a.exercises, ensemble_path=a.ensemble, classifier_path=a.classifier)
        print(HELP)
        buf: list[str] = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line.strip():
                buf.append(line)
                continue
            if buf:
                print(service.handle_text("\n".join(buf), chat_id="repl"))
                buf = []
        return 0
    token = os.environ.get("ARUZ_BOT_TOKEN")
    if not token:
        raise SystemExit("set ARUZ_BOT_TOKEN (from @BotFather on Telegram or the Bale bot builder)")
    poll(token, a.platform, a.lexicon, a.exercises, a.ensemble, a.classifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
