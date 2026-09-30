"""Minimal OpenAI-compatible chat client (works with OpenAI, Azure OpenAI proxies,
vLLM, Ollama, LiteLLM...). Kept dependency-free apart from httpx."""

from __future__ import annotations

import json
import os
import re
import time

import httpx


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_EVAL_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def chat(self, messages: list[dict], temperature: float = 0.0, json_mode: bool = False) -> str:
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"retryable status {resp.status_code}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, LLMError, KeyError) as err:
                last_err = err
                time.sleep(min(2**attempt, 8) * 0.25)
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_err}")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model response (tolerates ```json fences / prose)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in: {text[:200]!r}")
    return json.loads(match.group(0))
