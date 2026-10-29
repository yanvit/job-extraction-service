"""Локальна модель через Ollama REST API (/api/chat).

Structured output: параметр `format` = JSON Schema (Ollama >= 0.5 будує граматику з схеми,
тож модель фізично не може згенерувати невалідний за синтаксисом JSON). Семантику все одно валідуємо Pydantic-ом.
"""
from __future__ import annotations

import os
import time

import requests

from .base import FatalError, LLMClient, LLMResponse, RetryableError, Sampling, parse_json_loose


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str | None = None,
                 session: requests.Session | None = None, num_ctx: int = 8192,
                 think: bool | None = None, **kw):
        super().__init__(model, **kw)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx
        self.think = think  # для 'thinking' моделей (qwen3 тощо) — вимикаємо, щоб не ламати JSON
        self._http = session or requests.Session()

    def _call(self, system: str, user: str, schema: dict, sampling: Sampling) -> LLMResponse:
        options: dict = {"num_ctx": self.num_ctx, "num_predict": self.max_tokens}
        sent = {}
        if sampling.temperature is not None:
            options["temperature"] = sent["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            options["top_p"] = sent["top_p"] = sampling.top_p
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            body["think"] = self.think
        t0 = time.perf_counter()
        try:
            r = self._http.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
        except requests.Timeout as e:
            raise RetryableError(f"timeout after {self.timeout_s}s: {e!r}") from e
        except requests.ConnectionError as e:
            raise FatalError(f"Ollama is not reachable at {self.base_url} — run `ollama serve` "
                             f"and `ollama pull {self.model}`") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        if r.status_code >= 500 or r.status_code == 429:
            raise RetryableError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise FatalError(f"HTTP {r.status_code}: {r.text[:500]}")
        payload = r.json()
        text = payload.get("message", {}).get("content", "")
        return LLMResponse(
            raw_text=text,
            data=parse_json_loose(text),
            input_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            output_tokens=int(payload.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
            sampling_sent=sent,
            stop_reason=payload.get("done_reason"),
        )
