"""Провайдерська модель: Anthropic Claude через офіційний SDK.

- Транспорт — SDK `anthropic` з max_retries=0: ретраї робить наш спільний wrapper (base.with_retries),
  щоб політика ретраїв була однакова для Claude і для локальної моделі.
- Structured output — нативний: output_config.format = {type: json_schema, schema}.
  Якщо API/модель відхилить output_config (400), клієнт один раз перемикається на примусовий tool use.
- Семплінг: Haiku 5.5 не приймає temperature/top_p -> supports_sampling=false у config.yaml.
"""
from __future__ import annotations

import copy
import json
import os
import time

from .base import FatalError, LLMClient, LLMResponse, RetryableError, Sampling, parse_json_loose

TOOL_NAME = "record_job_posting"

# Обмеження, які structured outputs можуть не підтримувати. Їх перевіряє Pydantic після відповіді.
_UNSUPPORTED_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                     "minLength", "maxLength", "pattern", "format", "default", "title"}


def sanitize_schema(schema: dict) -> dict:
    """Прибирає ключові слова-обмеження і закриває об'єкти (additionalProperties: false)."""

    def walk(node, is_props_map=False):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if is_props_map:  # тут ключі — імена полів, а не ключові слова схеми
            return {k: walk(v) for k, v in node.items()}
        out = {}
        for k, v in node.items():
            if k in _UNSUPPORTED_KEYS:
                continue
            out[k] = walk(v, is_props_map=(k in ("properties", "$defs")))
        if out.get("type") == "object" and "properties" in out:
            out.setdefault("additionalProperties", False)
        return out

    return walk(copy.deepcopy(schema))


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, structured_mode: str = "json_schema",
                 supports_sampling: bool = True, sdk_client=None, **kw):
        super().__init__(model, **kw)
        if structured_mode not in ("auto", "json_schema", "tool"):
            raise ValueError(f"structured_mode must be auto|json_schema|tool, got {structured_mode!r}")
        # "auto" = почати з нативної JSON Schema і при відмові API один раз перейти на tool use
        self.structured_mode = "json_schema" if structured_mode == "auto" else structured_mode
        self.supports_sampling = supports_sampling
        self._drop_top_p = False
        if sdk_client is not None:                  # для тестів
            self._sdk = sdk_client
        else:
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise FatalError("ANTHROPIC_API_KEY is not set (copy .env.example to .env and fill it in)")
            import anthropic
            self._sdk = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=0)

    def _sampling_kwargs(self, s: Sampling) -> dict:
        if not self.supports_sampling:
            return {}
        out = {}
        if s.temperature is not None:
            out["temperature"] = s.temperature
        if s.top_p is not None and not (self._drop_top_p and s.temperature is not None):
            out["top_p"] = s.top_p
        return out

    def _call(self, system: str, user: str, schema: dict, sampling: Sampling) -> LLMResponse:
        import anthropic

        mode = self.structured_mode
        sent = self._sampling_kwargs(sampling)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        # temperature/top_p і output_config шлемо через extra_body: нові версії SDK прибрали
        # temperature/top_p з типізованих аргументів (моделі 5.x їх не приймають), а старі не знають output_config.
        extra_body: dict = dict(sent)
        if mode == "json_schema":
            extra_body["output_config"] = {"format": {"type": "json_schema", "schema": sanitize_schema(schema)}}
        else:
            kwargs["tools"] = [{"name": TOOL_NAME, "input_schema": schema,
                                "description": "Save the structured job posting extracted from the text."}]
            kwargs["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.perf_counter()
        try:
            msg = self._sdk.messages.create(**kwargs)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
            raise RetryableError(f"network/timeout: {e}") from e
        except anthropic.APIStatusError as e:
            code, text = e.status_code, str(e)
            if code in (429, 500, 502, 503, 504, 529):
                raise RetryableError(f"HTTP {code}: {text[:300]}") from e
            if code == 400 and "top_p" in sent and "top_p" in text and not self._drop_top_p:
                self._drop_top_p = True  # модель не приймає temperature і top_p разом
                raise RetryableError("top_p rejected together with temperature; retrying without it") from e
            if code == 400 and mode == "json_schema":
                self.structured_mode = "tool"
                raise RetryableError(f"output_config rejected, falling back to tool use: {text[:200]}") from e
            raise FatalError(f"HTTP {code}: {text[:500]}") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        data, raw = None, ""
        for block in msg.content:
            if mode == "tool" and block.type == "tool_use" and block.name == TOOL_NAME:
                data = block.input if isinstance(block.input, dict) else None
                raw = json.dumps(block.input, ensure_ascii=False)
                break
            if mode == "json_schema" and block.type == "text":
                raw = block.text
                data = parse_json_loose(raw)
                break
        return LLMResponse(
            raw_text=raw or str(msg.content)[:2000], data=data,
            input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
            latency_ms=latency_ms, sampling_sent={**sent, "structured_mode": mode},
            stop_reason=msg.stop_reason,
        )
