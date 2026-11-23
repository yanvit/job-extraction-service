"""Кеш відповідей LLM на диску: той самий (модель, system, user, схема) → та сама відповідь, без виклику API.

Навіщо: evals ганяються в CI на кожен push. Поки не змінились промпт, модель, питання чи знайдений контекст,
ключ кешу той самий — прогін безкоштовний і детермінований. Змінили щось — викликаються лише змінені пункти.
Ключ — sha256 від усього запиту; у файлі лише відповідь моделі й токени (текстів документів там немає).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from jobex.clients.base import LLMResponse


class CachedClient:
    def __init__(self, client, cache_dir: Path, read_only: bool = False):
        self.client, self.dir, self.read_only = client, Path(cache_dir), read_only
        self.model = getattr(client, "model", "unknown")
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def key(self, system: str, user: str, schema: dict) -> str:
        blob = json.dumps([self.model, system, user, schema], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def extract(self, system: str, user: str, schema: dict, sampling) -> LLMResponse:
        path = self.dir / f"{self.key(system, user, schema)}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            with self._lock:
                self.hits += 1
            return LLMResponse(raw_text=d["raw_text"], data=d["data"], input_tokens=d["input_tokens"],
                               output_tokens=d["output_tokens"], latency_ms=d["latency_ms"],
                               stop_reason=d.get("stop_reason"), sampling_sent={"cached": True})
        if self.read_only:
            raise RuntimeError(f"cache miss in read-only mode: {path.name} (model {self.model})")
        resp = self.client.extract(system, user, schema, sampling)
        with self._lock:
            self.misses += 1
        if resp.stop_reason != "max_tokens":            # обрізане не кешуємо — наступний прогін спробує знову
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "model": self.model, "raw_text": resp.raw_text, "data": resp.data,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "latency_ms": round(resp.latency_ms), "stop_reason": resp.stop_reason,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        return resp
