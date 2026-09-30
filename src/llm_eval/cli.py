"""Command line: ``llm-eval run`` (evaluate + gate) and ``llm-eval guard`` (check one text)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from .guardrails import GuardConfig, Guardrails
from .judge import HeuristicJudge, LLMJudge
from .llm import ChatClient
from .report import to_markdown, write_reports
from .runner import check_gate, load_dataset, load_json, run_eval
from .targets import LLMRAG, ExtractiveRAG, HallucinatingRAG, HTTPTarget, load_kb

app = typer.Typer(add_completion=False, help="Evaluate and guard LLM/RAG applications.")


def _make_target(name: str, kb_path: str, url: str | None):
    if name == "http":
        if not url:
            raise typer.BadParameter("--url is required for --target http")
        return HTTPTarget(url)
    kb = load_kb(kb_path)
    return {"extractive": ExtractiveRAG, "hallucinating": HallucinatingRAG, "llm": LLMRAG}[name](kb)


@app.command()
def run(
    dataset: str = typer.Option("data/golden.jsonl", help="Golden set (JSONL of EvalCase)."),
    kb: str = typer.Option("data/kb.jsonl", help="Knowledge base for the built-in RAG targets."),
    target: str = typer.Option("extractive", help="extractive | hallucinating | llm | http"),
    url: str = typer.Option(None, help="Endpoint for --target http."),
    judge: str = typer.Option("heuristic", help="heuristic | llm"),
    judge_model: str = typer.Option(None, help="Override judge model."),
    guards: bool = typer.Option(True, help="Apply input/output guardrails."),
    thresholds: str = typer.Option("baselines/thresholds.json", help="Absolute metric floors."),
    baseline: str = typer.Option(None, help="Previous *.summary.json to compare against."),
    tolerance: float = typer.Option(0.02, help="Allowed drop vs baseline."),
    name: str = typer.Option("eval", help="Run name (report file prefix)."),
    out: str = typer.Option("reports", help="Report directory."),
    update_baseline: bool = typer.Option(False, help="Write this run's aggregates to --baseline."),
) -> None:
    cases = load_dataset(dataset)
    tgt = _make_target(target, kb, url)
    jdg = LLMJudge(ChatClient(model=judge_model)) if judge == "llm" else HeuristicJudge()
    summary = run_eval(cases, tgt, jdg, Guardrails() if guards else None, run_name=name)

    failures = check_gate(
        summary,
        thresholds=load_json(thresholds) if thresholds and Path(thresholds).exists() else None,
        baseline=load_json(baseline) if baseline and Path(baseline).exists() else None,
        tolerance=tolerance,
    )
    md = write_reports(summary, out, failures)
    typer.echo(md.read_text(encoding="utf-8"))
    if update_baseline and baseline:
        shutil.copy(Path(out) / f"{name}.summary.json", baseline)
        typer.echo(f"baseline updated: {baseline}")
    if failures:
        raise typer.Exit(code=1)


@app.command()
def guard(
    text: str = typer.Argument(..., help="Text to check."),
    mode: str = typer.Option("input", help="input | output"),
    context: list[str] = typer.Option(None, help="Context passages (output mode)."),
    strict: bool = typer.Option(False, help="Block (not flag) ungrounded output."),
) -> None:
    g = Guardrails(GuardConfig(strict_grounding=strict))
    res = g.check_input(text) if mode == "input" else g.check_output(text, context or [])
    typer.echo(json.dumps(res.model_dump(mode="json"), indent=2))
    if res.action.value == "block":
        raise typer.Exit(code=2)


@app.command()
def show(path: str) -> None:
    """Re-render a saved <run>.json as markdown."""
    from .models import RunSummary

    typer.echo(to_markdown(RunSummary.model_validate_json(Path(path).read_text(encoding="utf-8"))))


if __name__ == "__main__":
    app()
