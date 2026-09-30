"""Run a target over a golden dataset, score it, and gate against thresholds/baselines."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from .guardrails import Guardrails
from .judge import Judge
from .metrics import compute_all
from .models import Action, CaseResult, EvalCase, JudgeVerdict, RunSummary
from .targets import Target


def load_dataset(path: str | Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        return [EvalCase.model_validate_json(line) for line in f if line.strip()]


def _run_case(case: EvalCase, target: Target, judge: Judge | None, guards: Guardrails | None) -> CaseResult:
    result = CaseResult(case_id=case.id, question=case.question, answer="", tags=case.tags)
    try:
        if guards:
            result.input_guard = guards.check_input(case.question)
            if result.input_guard.action == Action.BLOCK:
                result.answer = "[blocked by input guardrail]"
                # A blocked adversarial prompt counts as a correct refusal.
                result.metrics = {"refusal_correct": float(case.expect_refusal)}
                if judge:
                    result.judge = JudgeVerdict(
                        score=5 if case.expect_refusal else 1,
                        reason="blocked by input guardrail",
                        judge="guardrail",
                    )
                return result
        resp = target.answer(case.question)
        result.answer, result.latency_ms = resp.answer, resp.latency_ms
        result.metrics = compute_all(case, resp)
        if judge:
            result.judge = judge.judge(case, resp)
            result.metrics["judge_score"] = (result.judge.score - 1) / 4  # normalised 0..1
        if guards:
            result.output_guard = guards.check_output(resp.answer, resp.contexts)
    except Exception as err:  # noqa: BLE001 - one bad case must not kill the run
        result.error = f"{type(err).__name__}: {err}"
    return result


def _aggregate(results: list[CaseResult]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r.metrics.items():
            buckets[k].append(v)
    agg = {k: round(mean(v), 4) for k, v in buckets.items()}
    agg["error_rate"] = round(sum(r.error is not None for r in results) / max(len(results), 1), 4)
    lat = sorted(r.latency_ms for r in results if r.error is None)
    if lat:
        agg["latency_p95_ms"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2)
    return agg


def run_eval(
    cases: list[EvalCase],
    target: Target,
    judge: Judge | None = None,
    guards: Guardrails | None = None,
    run_name: str = "run",
    workers: int = 4,
) -> RunSummary:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: _run_case(c, target, judge, guards), cases))

    by_tag: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for t in r.tags:
            by_tag[t].append(r)

    guarded = [r for r in results if r.output_guard or r.input_guard]
    blocked = [
        r
        for r in guarded
        if (r.input_guard and r.input_guard.action == Action.BLOCK)
        or (r.output_guard and r.output_guard.action in (Action.BLOCK, Action.FLAG))
    ]
    return RunSummary(
        run_name=run_name,
        target=target.name,
        n_cases=len(results),
        aggregates=_aggregate(results),
        by_tag={t: _aggregate(rs) for t, rs in sorted(by_tag.items())},
        guard_block_rate=round(len(blocked) / max(len(guarded), 1), 4) if guarded else 0.0,
        cases=results,
    )


# ---------------------------------------------------------------- gating

LOWER_IS_BETTER = {"error_rate", "latency_p95_ms"}


def check_gate(
    summary: RunSummary,
    thresholds: dict[str, float] | None = None,
    baseline: dict[str, float] | None = None,
    tolerance: float = 0.02,
) -> list[str]:
    """Return human-readable failures. Empty list == pass.

    thresholds: absolute floors (or ceilings for lower-is-better metrics).
    baseline:   previous aggregates; any metric dropping by more than ``tolerance`` fails.
    """
    failures: list[str] = []
    agg = summary.aggregates
    for metric, limit in (thresholds or {}).items():
        if metric not in agg:
            continue
        val = agg[metric]
        bad = val > limit if metric in LOWER_IS_BETTER else val < limit
        if bad:
            op = "<=" if metric in LOWER_IS_BETTER else ">="
            failures.append(f"{metric}={val:.3f} violates threshold {op} {limit}")
    for metric, old in (baseline or {}).items():
        if metric not in agg or metric == "latency_p95_ms":
            continue
        new = agg[metric]
        delta = old - new if metric not in LOWER_IS_BETTER else new - old
        if delta > tolerance:
            failures.append(f"{metric} regressed {old:.3f} -> {new:.3f} (tolerance {tolerance})")
    return failures


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))
