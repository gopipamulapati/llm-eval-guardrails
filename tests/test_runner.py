import httpx

from llm_eval.guardrails import Guardrails
from llm_eval.judge import HeuristicJudge, LLMJudge
from llm_eval.llm import ChatClient, extract_json
from llm_eval.models import EvalCase, TargetResponse
from llm_eval.report import to_markdown
from llm_eval.runner import check_gate, run_eval
from llm_eval.targets import ExtractiveRAG, HallucinatingRAG

THRESHOLDS = {"faithfulness": 0.9, "refusal_correct": 0.9, "error_rate": 0.0}


def test_baseline_passes_gate(kb, cases):
    s = run_eval(cases, ExtractiveRAG(kb), HeuristicJudge(), Guardrails(), run_name="t")
    assert s.n_cases == len(cases)
    assert s.aggregates["faithfulness"] >= 0.9
    assert check_gate(s, THRESHOLDS) == []


def test_hallucinating_target_fails_gate_and_regresses(kb, cases):
    base = run_eval(cases, ExtractiveRAG(kb), HeuristicJudge(), Guardrails())
    bad = run_eval(cases, HallucinatingRAG(kb), HeuristicJudge(), Guardrails())
    failures = check_gate(bad, THRESHOLDS, baseline=base.aggregates)
    assert any("faithfulness" in f for f in failures)
    assert any("regressed" in f for f in failures)
    assert "FAIL" in to_markdown(bad, failures)


def test_adversarial_cases_blocked_before_target(kb, cases):
    s = run_eval(cases, ExtractiveRAG(kb), HeuristicJudge(), Guardrails())
    adv = [c for c in s.cases if "adversarial" in c.tags]
    assert adv and all(c.input_guard.action.value == "block" for c in adv)
    assert s.by_tag["adversarial"]["refusal_correct"] == 1.0


def test_errors_are_captured_not_raised(cases):
    class Broken:
        name = "broken"

        def answer(self, q):
            raise RuntimeError("boom")

    s = run_eval(cases[:3], Broken())
    assert s.aggregates["error_rate"] == 1.0
    assert check_gate(s, {"error_rate": 0.0})


def test_llm_judge_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": '```json\n{"score": 4, "reason": "ok"}\n```'}}]}
        return httpx.Response(200, json=body)

    client = ChatClient(model="mock", api_key="x", transport=httpx.MockTransport(handler))
    verdict = LLMJudge(client).judge(
        EvalCase(id="1", question="q", reference="r"), TargetResponse(answer="a", contexts=["c"])
    )
    assert verdict.score == 4 and verdict.judge == "llm:mock"


def test_extract_json_tolerates_prose():
    assert extract_json('Sure! {"score": 5, "reason": "x"} hope that helps') == {"score": 5, "reason": "x"}
