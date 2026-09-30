"""PII detection and redaction (regex + checksum validation, no external services)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Severity, Violation


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def ssn_valid(ssn: str) -> bool:
    area, group, serial = ssn.split("-")
    return area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000"


@dataclass(frozen=True)
class PIIPattern:
    kind: str
    regex: re.Pattern
    severity: Severity
    validator: object = None  # Callable[[str], bool] | None


PATTERNS: list[PIIPattern] = [
    PIIPattern("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), Severity.HIGH, ssn_valid),
    PIIPattern("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), Severity.HIGH, luhn_valid),
    PIIPattern("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), Severity.MEDIUM),
    PIIPattern(
        "phone",
        re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
        Severity.MEDIUM,
    ),
    PIIPattern(
        "bank_account",
        re.compile(r"\b(?:account|acct|routing)(?:\s+(?:no\.?|number|#))?\s*[:#]?\s*\d{6,17}\b", re.I),
        Severity.HIGH,
    ),
]


def find_pii(text: str) -> list[tuple[str, int, int, Severity]]:
    hits: list[tuple[str, int, int, Severity]] = []
    taken: list[tuple[int, int]] = []
    for pat in PATTERNS:
        for m in pat.regex.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            if pat.validator and not pat.validator(m.group(0)):  # type: ignore[operator]
                continue
            hits.append((pat.kind, s, e, pat.severity))
            taken.append((s, e))
    return sorted(hits, key=lambda h: h[1])


def redact(text: str) -> tuple[str, list[Violation]]:
    hits = find_pii(text)
    violations = [
        Violation(guard="pii", kind=kind, detail=f"{kind} at chars {s}-{e}", severity=sev) for kind, s, e, sev in hits
    ]
    for kind, s, e, _ in reversed(hits):
        text = text[:s] + f"[REDACTED_{kind.upper()}]" + text[e:]
    return text, violations
