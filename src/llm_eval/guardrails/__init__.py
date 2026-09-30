"""Composable input/output guardrails.

Policy:
  input  -> prompt-injection score >= block_threshold: BLOCK
            PII in the user prompt: REDACT before it reaches the model / logs
  output -> PII in the answer: REDACT
            unsupported claims > max_unsupported: FLAG (or BLOCK in strict mode)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Action, GuardResult
from .grounding import grounding_violations
from .injection import injection_score
from .pii import redact


@dataclass
class GuardConfig:
    block_threshold: float = 0.6
    redact_pii: bool = True
    max_unsupported: int = 0
    strict_grounding: bool = False


class Guardrails:
    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()

    def check_input(self, prompt: str) -> GuardResult:
        score, violations = injection_score(prompt)
        if score >= self.config.block_threshold:
            return GuardResult(action=Action.BLOCK, text="", violations=violations)
        text = prompt
        action = Action.FLAG if violations else Action.ALLOW
        if self.config.redact_pii:
            text, pii = redact(prompt)
            if pii:
                violations += pii
                action = Action.REDACT if action == Action.ALLOW else action
        return GuardResult(action=action, text=text, violations=violations)

    def check_output(self, answer: str, contexts: list[str] | None = None) -> GuardResult:
        text, violations = redact(answer) if self.config.redact_pii else (answer, [])
        action = Action.REDACT if violations else Action.ALLOW
        ground = grounding_violations(answer, contexts or [])
        if len(ground) > self.config.max_unsupported:
            violations += ground
            action = Action.BLOCK if self.config.strict_grounding else Action.FLAG
        return GuardResult(action=action, text=text if action != Action.BLOCK else "", violations=violations)


__all__ = ["GuardConfig", "Guardrails"]
