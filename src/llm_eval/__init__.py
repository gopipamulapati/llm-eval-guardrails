"""llm-eval-guardrails: evaluation harness + runtime guardrails for LLM/RAG apps."""

from .guardrails import GuardConfig, Guardrails
from .judge import HeuristicJudge, LLMJudge
from .models import EvalCase, RunSummary, TargetResponse
from .runner import check_gate, load_dataset, run_eval

__all__ = [
    "EvalCase",
    "GuardConfig",
    "Guardrails",
    "HeuristicJudge",
    "LLMJudge",
    "RunSummary",
    "TargetResponse",
    "check_gate",
    "load_dataset",
    "run_eval",
]
__version__ = "0.1.0"
