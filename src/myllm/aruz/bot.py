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
    "یک مصراع یا یک بیت (دو مصراع در دو خط) بفرست تا وزن، تقطیع هجایی، ارکان، اختیارات شاعری و قافیه‌اش را بگویم.\n\n"
    "مثال:\n"
    "بشنو این نی چون شکایت می‌کند\n"
    "از جدایی‌ها حکایت می‌کند\n\n"
    "⚠️ نسخه‌ی آزمایشی است؛ برای بیت‌های عربی، نیمایی و سپید وزن عروضی معنا ندارد یا ممکن است اشتباه کنم."
)
_SPLIT = re.compile(r"\s*(?:\n+|\s/\s|\*\*|\s\|\s)\s*")


class AruzService:
    def __init__(self, lexicon_path: str | None = None):
        self.scanner = Scanner(load_lexicon(lexicon_path))
        self.stats: Counter = Counter()

    def handle_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text or text.startswith("/start") or text.startswith("/help"):
            return HELP
        if text.startswith("/vazn"):
            text = text[len("/vazn"):].strip()
        parts = [p for p in _SPLIT.split(text) if p.strip()]
        if len(parts) > 2 or len(text) > 400:
            self.stats["too_long"] += 1
            return "لطفاً فقط یک بیت (حداکثر دو مصراع) بفرست."
        self.stats["requests"] += 1
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


def _call(base: str, token: str, method: str, params: dict, timeout: float = 70) -> dict:
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(f"{base}/bot{token}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def poll(token: str, platform: str = "bale", lexicon: str | None = None) -> None:  # pragma: no cover - network
    base = API_BASE[platform]
    service = AruzService(lexicon)
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
                reply = service.handle_text(msg["text"])
                _call(base, token, "sendMessage", {"chat_id": msg["chat"]["id"], "text": reply,
                                                   "reply_to_message_id": msg.get("message_id")})
        except Exception as exc:  # network hiccups: keep polling
            print("poll error:", exc)
            time.sleep(3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=sorted(API_BASE), default="bale")
    p.add_argument("--lexicon", default="data/processed/aruz_lexicon.json")
    p.add_argument("--repl", action="store_true", help="local console instead of a bot (no token needed)")
    a = p.parse_args(argv)
    if a.repl:
        service = AruzService(a.lexicon)
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
                print(service.handle_text("\n".join(buf)))
                buf = []
        return 0
    token = os.environ.get("ARUZ_BOT_TOKEN")
    if not token:
        raise SystemExit("set ARUZ_BOT_TOKEN (from @BotFather on Telegram or the Bale bot builder)")
    poll(token, a.platform, a.lexicon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
