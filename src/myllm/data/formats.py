"""Dataset format conversion (roadmap §4.4).

Everything is converted to the OpenAI/HF *conversational* format::

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}

which ``tokenizer.apply_chat_template`` and TRL understand natively.
"""

from __future__ import annotations

from typing import Any

_SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
}


def from_alpaca(row: dict[str, Any], system: str | None = None) -> dict:
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    output = (row.get("output") or row.get("response") or "").strip()
    user = f"{instruction}\n\n{inp}" if inp else instruction
    messages = [{"role": "system", "content": system}] if system else []
    messages += [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    return {"messages": messages}


def from_sharegpt(row: dict[str, Any]) -> dict:
    convs = row.get("conversations") or row.get("conversation") or []
    messages = []
    for turn in convs:
        role = _SHAREGPT_ROLES.get(str(turn.get("from", turn.get("role", ""))).lower())
        if role is None:
            raise ValueError(f"unknown ShareGPT role: {turn}")
        messages.append({"role": role, "content": (turn.get("value") or turn.get("content") or "").strip()})
    return {"messages": messages}


def to_messages(row: dict[str, Any], system: str | None = None) -> dict:
    """Auto-detect the row format and return ``{"messages": [...]}`` (+ passthrough metadata)."""
    if "messages" in row:
        out = {"messages": row["messages"]}
    elif "conversations" in row or "conversation" in row:
        out = from_sharegpt(row)
    elif "instruction" in row:
        out = from_alpaca(row, system=system)
    elif "prompt" in row and ("response" in row or "completion" in row):
        out = {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row.get("response") or row.get("completion")},
            ]
        }
    else:
        raise ValueError(f"cannot detect format of row with keys {sorted(row)}")
    for key in ("id", "source", "license", "lang", "weight"):
        if key in row:
            out[key] = row[key]
    return out


def validate_messages(messages: list[dict]) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    problems: list[str] = []
    if not messages:
        return ["empty conversation"]
    roles = [m.get("role") for m in messages]
    body = roles[1:] if roles[0] == "system" else roles
    if not body or body[0] != "user":
        problems.append("first non-system turn must be user")
    if "assistant" not in body:
        problems.append("no assistant turn to learn from")
    for a, b in zip(body, body[1:]):
        if a == b:
            problems.append(f"two consecutive {a} turns")
            break
    for i, m in enumerate(messages):
        if m.get("role") not in {"system", "user", "assistant", "tool"}:
            problems.append(f"turn {i}: bad role {m.get('role')!r}")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            problems.append(f"turn {i}: empty content")
    return problems


def conversation_text(messages: list[dict], roles: tuple[str, ...] = ("user", "assistant")) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") in roles)
