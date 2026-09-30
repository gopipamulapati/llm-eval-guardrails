"""Core data models shared by the harness, guardrails and reports."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One row of a golden dataset."""

    id: str
    question: str
    contexts: list[str] = Field(default_factory=list)
    reference: str | None = None
    # When True the correct behaviour is to refuse / say "I don't know".
    expect_refusal: bool = False
    tags: list[str] = Field(default_factory=list)


class TargetResponse(BaseModel):
    """What the system under test returned for one case."""

    answer: str
    contexts: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    FLAG = "flag"
    BLOCK = "block"


class Violation(BaseModel):
    guard: str
    kind: str
    detail: str
    severity: Severity = Severity.MEDIUM


class GuardResult(BaseModel):
    action: Action = Action.ALLOW
    text: str
    violations: list[Violation] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.action in (Action.ALLOW, Action.REDACT)


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str
    judge: str


class CaseResult(BaseModel):
    case_id: str
    question: str
    answer: str
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    judge: JudgeVerdict | None = None
    input_guard: GuardResult | None = None
    output_guard: GuardResult | None = None
    latency_ms: float = 0.0
    error: str | None = None


class RunSummary(BaseModel):
    run_name: str
    target: str
    n_cases: int
    aggregates: dict[str, float]
    by_tag: dict[str, dict[str, float]] = Field(default_factory=dict)
    guard_block_rate: float = 0.0
    cases: list[CaseResult] = Field(default_factory=list)
