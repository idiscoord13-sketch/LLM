"""LLM-as-a-judge with the bias controls of roadmap §8.2.

* Position bias  -> every pair is judged twice with A/B swapped; disagreement = tie.
* Verbosity bias -> the rubric says length is not quality; optional length-normalised win-rate.
* Self-preference -> configure a judge from a *different* family than the model under test.
* Calibration    -> ``calibration_agreement`` compares judge verdicts with human labels.

The judge is any client with ``.chat(messages, n) -> list[str]`` (see ``myllm.clients``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

PAIRWISE_PROMPT = """تو یک داور بی‌طرف و سخت‌گیر هستی. دو پاسخ به یک پرسش کاربر را مقایسه کن.
معیارها به ترتیب اهمیت: ۱) درستی و دقت (برای کد: آیا واقعاً کار می‌کند؟) ۲) پیروی کامل از دستور ۳) مفید بودن و عمق ۴) فارسی روان و درست ۵) ایمنی.
طولانی‌تر بودن به‌خودیِ خود امتیاز نیست. ترتیب نمایش پاسخ‌ها اهمیتی ندارد. به سبک نگارش یک مدل خاص امتیاز اضافه نده.
{reference_block}
[پرسش کاربر]
{prompt}

[پاسخ A]
{a}

[پاسخ B]
{b}

ابتدا در یکی دو جمله استدلال کن، سپس در خط آخر دقیقاً یکی از این‌ها را بنویس:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

POINTWISE_PROMPT = """تو یک داور سخت‌گیر هستی. پاسخ زیر را از ۱ تا ۱۰ نمره بده.
معیارها: درستی، پیروی از دستور، مفید بودن، فارسی روان، ایمنی. طول زیاد امتیاز نیست.
{reference_block}
[پرسش]
{prompt}

[پاسخ]
{answer}

در خط آخر فقط بنویس: SCORE: <عدد>"""

_VERDICT = re.compile(r"VERDICT:\s*(A|B|TIE)", re.IGNORECASE)
_SCORE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _reference_block(reference: str | None) -> str:
    return f"\n[پاسخ مرجع / راهنمای نمره‌دهی]\n{reference}\n" if reference else ""


def parse_verdict(text: str) -> str:
    matches = _VERDICT.findall(text or "")
    return matches[-1].upper() if matches else "TIE"


def parse_score(text: str) -> float | None:
    matches = _SCORE.findall(text or "")
    return float(matches[-1]) if matches else None


@dataclass
class PairResult:
    winner: str  # "A" | "B" | "tie"  (A = first argument of judge_pair)
    first: str
    second: str


def judge_pair(judge, prompt: str, a: str, b: str, reference: str | None = None,
               ask: Callable[[str], str] | None = None) -> PairResult:
    """Judge (a, b) and (b, a). A consistent winner across both orders wins; otherwise tie."""

    def _ask(x: str, y: str) -> str:
        text = PAIRWISE_PROMPT.format(prompt=prompt, a=x, b=y, reference_block=_reference_block(reference))
        if ask is not None:
            return ask(text)
        return judge.chat([{"role": "user", "content": text}], n=1, temperature=0.0)[0]

    v1 = parse_verdict(_ask(a, b))
    v2 = parse_verdict(_ask(b, a))
    if v1 == "A" and v2 == "B":
        winner = "A"
    elif v1 == "B" and v2 == "A":
        winner = "B"
    else:
        winner = "tie"
    return PairResult(winner, v1, v2)


def judge_score(judge, prompt: str, answer: str, reference: str | None = None) -> float | None:
    text = POINTWISE_PROMPT.format(prompt=prompt, answer=answer, reference_block=_reference_block(reference))
    return parse_score(judge.chat([{"role": "user", "content": text}], n=1, temperature=0.0)[0])


def win_rate(results: list[str]) -> float:
    """Ties count as half a win (standard in arena-style evaluation)."""
    if not results:
        return 0.0
    return (sum(r == "A" for r in results) + 0.5 * sum(r == "tie" for r in results)) / len(results)


def calibration_agreement(judge_verdicts: list[str], human_verdicts: list[str]) -> float:
    """Fraction of items where the judge agrees with a human label (A/B/tie)."""
    pairs = list(zip(judge_verdicts, human_verdicts))
    return sum(j.lower() == h.lower() for j, h in pairs) / max(1, len(pairs))
