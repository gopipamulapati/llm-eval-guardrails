"""Prompt-injection / jailbreak detection with weighted heuristics.

Heuristics are a first line of defence: fast, explainable, and easy to unit test.
In production pair them with a classifier model; the interface stays the same.
"""

from __future__ import annotations

import base64
import re

from ..models import Severity, Violation

RULES: list[tuple[str, float, re.Pattern]] = [
    (
        "override_instructions",
        0.6,
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b"
            r".{0,20}\b(instructions?|prompts?|rules?|directions?)",
            re.I,
        ),
    ),
    (
        "reveal_system_prompt",
        0.6,
        re.compile(
            r"\b(reveal|show|print|repeat|output|leak)\b.{0,40}\b(system|hidden|initial)\s+"
            r"(prompt|instructions?|message)",
            re.I,
        ),
    ),
    ("role_hijack", 0.4, re.compile(r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b", re.I)),
    ("jailbreak_persona", 0.5, re.compile(r"\b(DAN|developer mode|jailbreak|no restrictions)\b", re.I)),
    (
        "delimiter_spoof",
        0.4,
        re.compile(r"(</?(system|assistant|instructions?)>|\[/?INST\]|<\|im_start\|>|###\s*system)", re.I),
    ),
    (
        "exfiltration",
        0.5,
        re.compile(r"\b(send|post|upload|exfiltrate)\b.{0,40}\b(https?://|webhook|api key|password|credentials)", re.I),
    ),
]

_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")


def _decoded_payloads(text: str) -> list[str]:
    out = []
    for m in _B64_RE.finditer(text):
        try:
            decoded = base64.b64decode(m.group(0), validate=True).decode("utf-8")
        except Exception:  # noqa: BLE001 - not base64 text, ignore
            continue
        if decoded.isprintable():
            out.append(decoded)
    return out


def injection_score(text: str) -> tuple[float, list[Violation]]:
    """Return a 0..1 risk score and the rules that fired (also scans base64 payloads)."""
    violations: list[Violation] = []
    score = 0.0
    for source, body in [("text", text)] + [("base64", p) for p in _decoded_payloads(text)]:
        for name, weight, rx in RULES:
            if rx.search(body):
                bonus = 0.2 if source == "base64" else 0.0  # hiding it is itself suspicious
                score += weight + bonus
                sev = Severity.HIGH if weight >= 0.5 else Severity.MEDIUM
                violations.append(Violation(guard="injection", kind=name, detail=f"matched in {source}", severity=sev))
    return min(score, 1.0), violations
