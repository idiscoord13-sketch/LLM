"""Test-time compute: spend more inference to get a smarter answer from a small model.

Given n sampled candidates, pick one:
  * code   — prefer candidates whose code actually runs (syntax + top-level execution),
             then the most self-consistent among them;
  * math   — majority vote on the final number (self-consistency, Wang et al. 2022);
  * other  — the candidate most similar to the others (a "medoid"; robust to outliers).

Cheap to add at serving time, often worth several model sizes on reasoning/code.
"""

from __future__ import annotations

from myllm.eval.code_exec import extract_code, run_python
from myllm.research.self_consistency import agreement_scores, final_answer


def classify(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ("python", "پایتون", "کد", "تابع", "function", "```", "برنامه")):
        return "code"
    if any(ch.isdigit() for ch in p) and any(k in p for k in ("چند", "حساب", "how many", "calculate", "=", "+", "×", "*")):
        return "math"
    return "other"


def runs_ok(text: str, timeout: float = 3.0) -> bool:
    code = extract_code(text)
    if not code.strip():
        return False
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError:
        return False
    return run_python(code, timeout=timeout).passed


def select_best(prompt: str, candidates: list[str], kind: str | None = None) -> tuple[int, dict]:
    if len(candidates) == 1:
        return 0, {"kind": kind or "single"}
    kind = kind or classify(prompt)
    agree, uncertainty = agreement_scores(candidates)
    info = {"kind": kind, "uncertainty": round(uncertainty, 3)}
    if kind == "code":
        ok = [runs_ok(c) for c in candidates]
        info["runs_ok"] = ok
        pool = [i for i, good in enumerate(ok) if good] or list(range(len(candidates)))
        return max(pool, key=lambda i: agree[i]), info
    if kind == "math":
        answers = [final_answer(c) for c in candidates]
        info["answers"] = answers
    return max(range(len(candidates)), key=lambda i: agree[i]), info
