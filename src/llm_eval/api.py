"""Guardrails as a sidecar service: call /guard/input before your LLM and /guard/output after.

uvicorn llm_eval.api:app --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .guardrails import GuardConfig, Guardrails
from .models import GuardResult

app = FastAPI(title="LLM Guardrails", version="0.1.0")
_guards = Guardrails(GuardConfig())


class InputReq(BaseModel):
    prompt: str


class OutputReq(BaseModel):
    answer: str
    contexts: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/guard/input", response_model=GuardResult)
def guard_input(req: InputReq) -> GuardResult:
    return _guards.check_input(req.prompt)


@app.post("/guard/output", response_model=GuardResult)
def guard_output(req: OutputReq) -> GuardResult:
    return _guards.check_output(req.answer, req.contexts)
