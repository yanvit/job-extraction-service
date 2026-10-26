"""Спільний інтерфейс для провайдерської та локальної моделей."""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Sampling:
    """Параметри семплінгу. None = не передавати параметр (дефолт моделі)."""

    name: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def as_dict(self) -> dict:
        return {"name": self.name, "temperature": self.temperature, "top_p": self.top_p}


@dataclass
class LLMResponse:
    raw_text: str                       # те, що повернула модель (JSON-рядок)
    data: Optional[dict]                # розпарсений JSON або None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0             # чистий час останнього успішного HTTP-виклику
    attempts: int = 1                   # скільки HTTP-спроб знадобилось (ретраї)
    sampling_sent: dict = field(default_factory=dict)  # що реально пішло в API
    stop_reason: Optional[str] = None


class RetryableError(Exception):
    """Тимчасова помилка (таймаут, 429, 5xx) — варто повторити."""


class FatalError(Exception):
    """Помилка, яку ретрай не виправить (400, 401, ...)."""


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str, timeout_s: float = 60.0, max_retries: int = 3,
                 backoff_base_s: float = 1.0, max_tokens: int = 2048):
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.max_tokens = max_tokens

    @abstractmethod
    def _call(self, system: str, user: str, schema: dict, sampling: Sampling) -> LLMResponse:
        """Один HTTP-виклик. Кидає RetryableError / FatalError."""

    def extract(self, system: str, user: str, schema: dict, sampling: Sampling) -> LLMResponse:
        """Виклик з ретраями та експоненційним backoff (+jitter)."""
        return with_retries(lambda: self._call(system, user, schema, sampling),
                            self.max_retries, self.backoff_base_s)


def with_retries(fn: Callable[[], LLMResponse], max_retries: int, backoff_base_s: float,
                 sleep: Callable[[float], Any] = time.sleep) -> LLMResponse:
    last: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):  # 1 спроба + max_retries повторів
        try:
            resp = fn()
            resp.attempts = attempt
            return resp
        except RetryableError as e:
            last = e
            if attempt > max_retries:
                break
            delay = backoff_base_s * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            sleep(delay)
    raise RetryableError(f"gave up after {max_retries + 1} attempts: {last}")


def parse_json_loose(text: str) -> Optional[dict]:
    """Дістає JSON-об'єкт з тексту (на випадок ```json ...``` чи сміття навколо)."""
    import json

    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None
