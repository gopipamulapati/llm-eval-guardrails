"""Markdown + JSON reports (the markdown is designed to be posted as a PR comment)."""

from __future__ import annotations

import json
from pathlib import Path

from .models import RunSummary

KEY_METRICS = ["judge_score", "faithfulness", "token_f1", "rouge_l", "context_recall", "refusal_correct"]


def to_markdown(summary: RunSummary, failures: list[str] | None = None) -> str:
    agg = summary.aggregates
    status = "PASS" if not failures else "FAIL"
    lines = [
        f"## LLM eval: `{summary.run_name}` - **{status}**",
        "",
        f"Target: `{summary.target}` | cases: {summary.n_cases} | "
        f"guardrail block/flag rate: {summary.guard_block_rate:.0%}",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in KEY_METRICS + [k for k in agg if k not in KEY_METRICS]:
        if k in agg:
            lines.append(f"| {k} | {agg[k]:.3f} |")
    if summary.by_tag:
        cols = [k for k in KEY_METRICS if any(k in v for v in summary.by_tag.values())]
        lines += ["", "### By tag", "", "| tag | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for tag, vals in summary.by_tag.items():
            lines.append(f"| {tag} | " + " | ".join(f"{vals[c]:.2f}" if c in vals else "-" for c in cols) + " |")
    if failures:
        lines += ["", "### Gate failures", ""] + [f"- {f}" for f in failures]
    worst = sorted(
        (c for c in summary.cases if c.judge), key=lambda c: (c.judge.score, c.metrics.get("faithfulness", 1))
    )[:5]
    if worst:
        lines += ["", "### Lowest-scoring cases", "", "| id | judge | reason | answer |", "|---|---|---|---|"]
        for c in worst:
            ans = c.answer.replace("|", "/").replace("\n", " ")[:90]
            lines.append(f"| {c.case_id} | {c.judge.score} | {c.judge.reason[:60]} | {ans} |")
    return "\n".join(lines) + "\n"


def write_reports(summary: RunSummary, out_dir: str | Path, failures: list[str] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{summary.run_name}.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (out / f"{summary.run_name}.summary.json").write_text(json.dumps(summary.aggregates, indent=2), encoding="utf-8")
    md = out / f"{summary.run_name}.md"
    md.write_text(to_markdown(summary, failures), encoding="utf-8")
    return md
