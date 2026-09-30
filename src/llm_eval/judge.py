"""LLM-as-judge with a fixed rubric, plus an offline heuristic judge.

The heuristic judge lets the whole pipeline (and CI) run with no API key; swap in
``LLMJudge`` for the real signal. Both return the same ``JudgeVerdict``.
"""

from __future__ import annotations

from typing import Protocol

from .llm import ChatClient, extract_json
from .metrics import faithfulness, is_refusal, token_f1
from .models import EvalCase, JudgeVerdict, TargetResponse

RUBRIC = """You are a strict evaluator of answers produced by a retrieval-augmented assistant.
Score the ANSWER from 1 to 5:
5 = fully correct, complete, and every claim is supported by the CONTEXT
4 = correct and grounded, minor omission
3 = partially correct or contains a minor unsupported claim
2 = mostly incorrect, or a significant unsupported (hallucinated) claim
1 = wrong, off-topic, or unsafe
If the context does not contain the answer, the best answer says so; reward correct refusals
with 5 and penalize guessing.
Return ONLY JSON: {"score": <int 1-5>, "reason": "<one sentence>"}"""


class Judge(Protocol):
    name: str

    def judge(self, case: EvalCase, resp: TargetResponse) -> JudgeVerdict: ...


class LLMJudge:
    def __init__(self, client: ChatClient | None = None) -> None:
        self.client = client or ChatClient()
        self.name = f"llm:{self.client.model}"

    def judge(self, case: EvalCase, resp: TargetResponse) -> JudgeVerdict:
        context = "\n---\n".join(resp.contexts) or "(no context retrieved)"
        user = (
            f"QUESTION:\n{case.question}\n\nCONTEXT:\n{context}\n\n"
            f"REFERENCE ANSWER (may be empty):\n{case.reference or ''}\n\nANSWER:\n{resp.answer}"
        )
        raw = self.client.chat(
            [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}],
            json_mode=True,
        )
        data = extract_json(raw)
        score = max(1, min(5, int(data.get("score", 1))))
        return JudgeVerdict(score=score, reason=str(data.get("reason", ""))[:500], judge=self.name)


class HeuristicJudge:
    """Deterministic stand-in for an LLM judge, built from the cheap metrics."""

    name = "heuristic"

    def judge(self, case: EvalCase, resp: TargetResponse) -> JudgeVerdict:
        refused = is_refusal(resp.answer)
        if case.expect_refusal:
            if refused:
                return JudgeVerdict(score=5, reason="Correctly declined: answer not in context.", judge=self.name)
            return JudgeVerdict(score=1, reason="Answered a question that should have been declined.", judge=self.name)
        if refused:
            return JudgeVerdict(score=2, reason="Declined although the answer was available.", judge=self.name)

        faith = faithfulness(resp.answer, resp.contexts)
        f1 = token_f1(resp.answer, case.reference) if case.reference else faith
        blended = 0.5 * faith + 0.5 * f1
        score = 1 + round(4 * blended)
        reason = f"faithfulness={faith:.2f}, token_f1={f1:.2f}"
        return JudgeVerdict(score=score, reason=reason, judge=self.name)
