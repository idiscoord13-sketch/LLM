"""Execute model-written Python against unit tests (verifiable reward for eval + GRPO).

    passed, info = run_tests(completion_text, "assert add(2, 3) == 5")

Isolation: a fresh ``python -I`` subprocess in a temp dir, wall-clock timeout, and on
POSIX CPU/memory/file-size rlimits. This is a *guard rail*, not a security sandbox —
for untrusted public traffic run it inside a container/VM (e.g. Docker with
``--network none --read-only``) or a dedicated sandbox service.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the last fenced python block, else the whole text (minus any <think> block)."""
    text = _THINK.sub("", text or "")
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    return text.strip()


@dataclass
class ExecResult:
    passed: bool
    returncode: int | None
    timeout: bool
    stdout: str
    stderr: str


def _limits(memory_mb: int, cpu_seconds: int):  # pragma: no cover - runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 2**20, memory_mb * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20, 10 * 2**20))


def run_python(source: str, timeout: float = 5.0, memory_mb: int = 512) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="myllm_exec_") as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        preexec = None
        if os.name == "posix":
            cpu = max(1, int(timeout) + 1)
            preexec = lambda: _limits(memory_mb, cpu)  # noqa: E731
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            return ExecResult(proc.returncode == 0, proc.returncode, False, proc.stdout[-2000:], proc.stderr[-2000:])
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return ExecResult(False, None, True, out[-2000:], "timeout")


def run_tests(completion: str, tests: str, timeout: float = 5.0) -> tuple[bool, ExecResult]:
    code = extract_code(completion)
    result = run_python(code + "\n\n" + tests + "\n", timeout=timeout)
    return result.passed, result


def pass_fraction(completion: str, tests: str, timeout: float = 5.0) -> float:
    """Fraction of assert lines that pass (denser reward than all-or-nothing)."""
    asserts = [ln for ln in tests.splitlines() if ln.strip()]
    if not asserts:
        return 0.0
    code = extract_code(completion)
    harness = [code, "", "_passed = 0"]
    for line in asserts:
        harness += ["try:", f"    {line}", "    _passed += 1", "except Exception:", "    pass"]
    harness.append("print('__PASSED__', _passed)")
    result = run_python("\n".join(harness) + "\n", timeout=timeout)
    m = re.search(r"__PASSED__ (\d+)", result.stdout)
    return int(m.group(1)) / len(asserts) if m else 0.0
