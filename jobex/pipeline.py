"""Прогін документів: виклик моделі -> валідація Pydantic -> 1 спроба виправлення -> лог результату."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import ValidationError

from .clients import FatalError, LLMClient, RetryableError, Sampling
from .prompts import Prompt
from .schema import JobPosting, inline_refs, json_schema


@dataclass
class Doc:
    id: str
    text: str


def load_docs(data_dir: Path, only_ids: Optional[Iterable[str]] = None) -> list[Doc]:
    ids = set(only_ids) if only_ids is not None else None
    docs = [Doc(p.stem, p.read_text(encoding="utf-8")) for p in sorted(Path(data_dir).glob("*.txt"))]
    return [d for d in docs if ids is None or d.id in ids]


def format_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


def load_golden_ids(path: Path) -> list[str]:
    return [json.loads(l)["doc_id"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def validate(data: Optional[dict]) -> tuple[Optional[JobPosting], Optional[str]]:
    if data is None:
        return None, "- <root>: response is not a JSON object"
    try:
        return JobPosting.model_validate(data), None
    except ValidationError as e:
        return None, format_errors(e)


def schema_for(client: LLMClient) -> dict:
    s = json_schema()
    return inline_refs(s) if client.provider == "ollama" else s


def cache_key(model_id: str, model: str, prompt: Prompt, sampling: Sampling, text: str) -> str:
    payload = json.dumps({"m": model_id, "model": model, "p": prompt.version, "psha": prompt.sha,
                          "s": sampling.as_dict(), "t": text,
                          "schema": hashlib.sha256(json.dumps(json_schema(), sort_keys=True).encode()).hexdigest()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def extract_one(client: LLMClient, model_cfg: dict, prompt: Prompt, sampling: Sampling, doc: Doc,
                run_id: str) -> dict:
    """Повертає запис логу для одного документа. Ніколи не кидає виняток назовні."""
    schema = schema_for(client)
    rec = {
        "run_id": run_id, "doc_id": doc.id,
        "model_id": model_cfg["id"], "provider": client.provider, "model": client.model,
        "prompt_version": prompt.version, "prompt_sha": prompt.sha,
        "sampling": sampling.as_dict(), "sampling_sent": None,
        "status": None, "repair_attempted": False, "http_attempts": 0,
        "latency_ms": 0.0, "first_call_latency_ms": None,
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        "output": None, "raw_first": None, "validation_errors": None, "error": None,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    price = model_cfg.get("price_per_mtok", {"input": 0, "output": 0})
    t_start = time.perf_counter()

    def account(resp):
        rec["http_attempts"] += resp.attempts
        rec["input_tokens"] += resp.input_tokens
        rec["output_tokens"] += resp.output_tokens
        rec["sampling_sent"] = resp.sampling_sent

    try:
        resp = client.extract(prompt.system, prompt.render_user(doc.text), schema, sampling)
        account(resp)
        rec["first_call_latency_ms"] = round(resp.latency_ms, 1)
        rec["raw_first"] = resp.raw_text[:4000]
        obj, errors = validate(resp.data)
        if obj is not None:
            rec["status"], rec["output"] = "ok", obj.model_dump(mode="json")
        else:
            # одна спроба виправлення з текстом помилок валідації
            rec["repair_attempted"] = True
            rec["validation_errors"] = errors
            fix = client.extract(prompt.system, prompt.render_repair(doc.text, resp.raw_text[:6000], errors),
                                 schema, sampling)
            account(fix)
            obj2, errors2 = validate(fix.data)
            if obj2 is not None:
                rec["status"], rec["output"] = "repaired", obj2.model_dump(mode="json")
            else:
                rec["status"], rec["validation_errors"] = "failed", errors2
    except (FatalError, RetryableError) as e:
        rec["status"], rec["error"] = "failed", f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 — один зламаний документ не валить весь прогін
        rec["status"], rec["error"] = "failed", f"Unexpected {type(e).__name__}: {e}"

    rec["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    rec["cost_usd"] = round(rec["input_tokens"] / 1e6 * price["input"] + rec["output_tokens"] / 1e6 * price["output"], 8)
    return rec


def run_name(model_id: str, prompt_version: str, sampling_name: str) -> str:
    return f"{model_id}__{prompt_version}__{sampling_name}"


def run_experiment(client: LLMClient, model_cfg: dict, prompt: Prompt, sampling: Sampling, docs: list[Doc],
                   results_dir: Path, use_cache: bool = True, concurrency: int = 1,
                   log=print) -> list[dict]:
    results_dir = Path(results_dir)
    cache_dir = results_dir / "cache"
    runs_dir = results_dir / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    name = run_name(model_cfg["id"], prompt.version, sampling.name)

    def process(doc: Doc) -> dict:
        key = cache_key(model_cfg["id"], client.model, prompt, sampling, doc.text)
        cpath = cache_dir / f"{key}.json"
        if use_cache and cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
            rec["cached"] = True
            return rec
        rec = extract_one(client, model_cfg, prompt, sampling, doc, run_id)
        rec["cached"] = False
        if rec["status"] != "failed" or rec.get("validation_errors"):
            # кешуємо результати моделі (включно з невалідними), але не інфраструктурні збої
            cpath.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec

    records: list[dict] = []
    fatal_streak = 0
    log(f"  {name}: {len(docs)} docs ...")

    def progress(i: int, rec: dict) -> None:
        src = "cache" if rec.get("cached") else f'{rec["latency_ms"] / 1000:.1f}s'
        log(f"    [{i}/{len(docs)}] {rec['doc_id']} {rec['status']} ({src})")

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for i, rec in enumerate(pool.map(process, docs), 1):
                records.append(rec)
                progress(i, rec)
    else:
        for i, doc in enumerate(docs, 1):
            rec = process(doc)
            records.append(rec)
            progress(i, rec)
            fatal_streak = fatal_streak + 1 if (rec["error"] or "").startswith("FatalError") else 0
            if fatal_streak >= 3:
                log(f"  ! {name}: 3 fatal errors in a row ({rec['error'][:120]}) — skipping the rest")
                break

    out = runs_dir / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(r["status"] in ("ok", "repaired") for r in records)
    log(f"  {name}: {n_ok}/{len(records)} valid -> {out}")
    return records
