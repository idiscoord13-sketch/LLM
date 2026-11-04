"""License gate for training data (roadmap §14.2).

``commercial_safe: true`` drops rows whose license forbids commercial use and warns
about unknown licenses. This is engineering hygiene, not legal advice — read each
dataset card yourself (licenses change, and derived datasets inherit restrictions,
e.g. anything translated from Stanford Alpaca is CC BY-NC).
"""

from __future__ import annotations

NON_COMMERCIAL = ("cc-by-nc", "nc-", "-nc", "non-commercial", "noncommercial", "research-only", "research only")
PERMISSIVE = ("apache-2.0", "mit", "bsd", "cc0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa", "odc-by", "odc-by-1.0",
              "pddl", "unlicense", "llama", "gemma", "qwen")


def classify(license_id: str | None) -> str:
    """Return 'commercial_ok' | 'non_commercial' | 'unknown'."""
    if not license_id:
        return "unknown"
    lic = license_id.strip().lower()
    if any(tag in lic for tag in NON_COMMERCIAL):
        return "non_commercial"
    if any(lic.startswith(p) or lic == p for p in PERMISSIVE):
        return "commercial_ok"
    return "unknown"


def allowed(license_id: str | None, commercial_safe: bool = True, allow_unknown: bool = True) -> bool:
    kind = classify(license_id)
    if not commercial_safe:
        return True
    if kind == "non_commercial":
        return False
    return kind == "commercial_ok" or allow_unknown
