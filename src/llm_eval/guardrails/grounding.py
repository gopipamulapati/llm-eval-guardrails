"""Output grounding check: flag answer sentences not supported by retrieved context."""

from __future__ import annotations

from ..metrics import is_refusal, sentence_supported
from ..models import Severity, Violation
from ..text import sentences


def unsupported_sentences(answer: str, contexts: list[str], threshold: float = 0.6) -> list[str]:
    if is_refusal(answer) or not contexts:
        return []
    return [s for s in sentences(answer) if not sentence_supported(s, contexts, threshold)]


def grounding_violations(answer: str, contexts: list[str]) -> list[Violation]:
    return [
        Violation(guard="grounding", kind="unsupported_claim", detail=s[:200], severity=Severity.MEDIUM)
        for s in unsupported_sentences(answer, contexts)
    ]
