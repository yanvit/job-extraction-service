"""RAG з джерелами (ДЗ2, крок 3): пошук → контекст у межах бюджету → відповідь моделі з цитатами.

    python -m rag.answer "Який річний ліміт доходу ФОП 3 групи у 2026 році?"
    python -m rag.answer "Які Java-вакансії вимагають Kafka?" --doc-type vacancy --stack Java
    python -m rag.answer "..." --json        # повний результат: джерела, токени, вартість, версія промпту

«Не знаю» на двох рівнях:
  1) retrieval gate: найкращий чанк після reranker-а слабший за answer.min_rerank_score → not_found
     без виклику моделі (дешево, детерміновано);
  2) модель: промпт вимагає status=not_found / partial, коли в джерелах відповіді немає або вона неповна.
Після відповіді код перевіряє цитати: твердження без жодного джерела з контексту викидається.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass

import yaml

from .chunking import approx_tokens
from .index import CFG, ROOT

PROMPTS_DIR = ROOT / "prompts" / "rag"
ACFG = CFG["answer"]

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "not_found"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "sources"],
            },
        },
        "missing": {"type": ["string", "null"]},
    },
    "required": ["status", "claims", "missing"],
}

NOT_FOUND_TEXT = "У документах корпусу відповіді на це питання немає."


@dataclass(frozen=True)
class RagPrompt:
    version: str
    system: str
    user_template: str
    meta: dict

    @property
    def sha(self) -> str:
        return hashlib.sha256((self.system + self.user_template).encode()).hexdigest()[:12]

    def render(self, sources: str, question: str) -> str:
        return self.user_template.replace("{{SOURCES}}", sources).replace("{{QUESTION}}", question)


def load_rag_prompt(version: str) -> RagPrompt:
    d = PROMPTS_DIR / version
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8")) or {}
    if meta.get("version") != version:
        raise ValueError(f"{d}/meta.yaml: version must be '{version}'")
    return RagPrompt(version, (d / "system.md").read_text(encoding="utf-8").strip(),
                     (d / "user.md").read_text(encoding="utf-8").strip(), meta)


def clean_section(section: str | None) -> str:
    """У розділах HTML трапляється «заголовок | заголовок > підрозділ» — лишаємо останній, найповніший шлях."""
    return (section or "").split(" | ")[-1].strip()


def source_block(sid: str, chunk: dict) -> str:
    head = [f"[{sid}] Документ: {chunk['title']}"]
    if clean_section(chunk.get("section")):
        head.append(f"Розділ: {clean_section(chunk['section'])}")
    head.append(f"Версія: {chunk.get('version') or '—'} · тип: {chunk['doc_type']}")
    return "\n".join(head) + "\n" + chunk["text"].strip()


def pack_context(hits: list, budget_tokens: int, min_score: float) -> tuple[list[dict], int]:
    """Джерела в порядку релевантності, поки влазять у бюджет. Чанк, що не влазить, пропускаємо
    (наступний коротший може влізти); слабкі за min_score не беремо зовсім."""
    sources, used = [], 0
    for h in hits:
        if h.score < min_score:
            continue
        sid = f"S{len(sources) + 1}"
        block = source_block(sid, h.chunk)
        t = approx_tokens(block)
        if used + t > budget_tokens:
            continue
        sources.append({"sid": sid, "chunk": h.chunk, "score": round(h.score, 3), "tokens": t, "block": block})
        used += t
    return sources, used


def validate(data: dict | None, sids: set[str]) -> tuple[dict, list[dict]]:
    """Лишає твердження з ≥1 валідним джерелом; повертає (чиста відповідь, викинуті твердження)."""
    if not data or data.get("status") not in ("answered", "partial", "not_found"):
        return {"status": "not_found", "claims": [], "missing": "некоректна відповідь моделі"}, []
    kept, dropped = [], []
    for c in data.get("claims") or []:
        src = [s for s in (c.get("sources") or []) if s in sids]
        (kept if src and (c.get("text") or "").strip() else dropped).append({"text": c.get("text", ""), "sources": src})
    status = data["status"]
    if status != "not_found" and not kept:
        status = "not_found"
    if status == "not_found":
        kept = []
    return {"status": status, "claims": kept, "missing": data.get("missing")}, dropped


def _model_cfg(model_id: str) -> tuple[dict, dict]:
    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = next(m for m in root["models"] if m["id"] == model_id)
    return cfg, root.get("defaults", {})


class Answerer:
    def __init__(self, retriever=None, client=None, prompt_version: str | None = None, model_id: str | None = None):
        self.model_id = model_id or ACFG["model"]
        self.model_cfg, self._defaults = _model_cfg(self.model_id)
        self.prompt = load_rag_prompt(prompt_version or ACFG["prompt"])
        self._retriever, self._client = retriever, client

    @property
    def retriever(self):
        if self._retriever is None:
            from .search import Retriever
            self._retriever = Retriever("structure")
        return self._retriever

    @property
    def client(self):
        if self._client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            from jobex.clients import make_client
            self._client = make_client({**self.model_cfg, "max_tokens": ACFG["max_tokens"]}, self._defaults)
        return self._client

    def cost(self, tin: int, tout: int) -> float:
        p = self.model_cfg.get("price_per_mtok", {})
        return (tin * p.get("input", 0) + tout * p.get("output", 0)) / 1e6

    def answer(self, question: str, flt: dict | None = None) -> dict:
        from jobex.clients.base import Sampling

        t0 = time.perf_counter()
        hits = self.retriever.search(question, ACFG["retrieve_mode"], k=ACFG["retrieve_k"], flt=flt)
        top = hits[0].score if hits else float("-inf")
        base = {"question": question, "filter": flt, "model": self.model_id, "prompt_version": self.prompt.version,
                "prompt_sha": self.prompt.sha, "top_score": round(top, 3),
                "retrieved": [{"chunk_id": h.chunk["chunk_id"], "score": round(h.score, 3)} for h in hits]}

        if top < ACFG["min_rerank_score"]:
            return {**base, "status": "not_found", "claims": [], "dropped_claims": [], "gate": "retrieval",
                    "missing": f"жоден фрагмент не релевантний (найкращий бал {top:.2f} < {ACFG['min_rerank_score']})",
                    "sources": [], "context_tokens": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    "latency_ms": round(1000 * (time.perf_counter() - t0))}

        sources, used = pack_context(hits, ACFG["context_budget_tokens"], ACFG["min_source_score"])
        user = self.prompt.render("\n\n".join(s["block"] for s in sources), question)
        resp = self.client.extract(self.prompt.system, user, SCHEMA, Sampling("api_default"))
        clean, dropped = validate(resp.data, {s["sid"] for s in sources})
        if resp.stop_reason == "max_tokens":          # обрізаний JSON — це збій, а не чесне «не знаю»
            clean["missing"] = "відповідь моделі обрізано (max_tokens)"
            gate = "error"
        else:
            gate = "model" if clean["status"] == "not_found" else None
        return {**base, **clean, "dropped_claims": dropped, "gate": gate, "stop_reason": resp.stop_reason,
                "sources": [{k: v for k, v in s.items() if k != "block"} | {"chunk": _cite(s["chunk"])} for s in sources],
                "context_tokens": used, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "cost_usd": round(self.cost(resp.input_tokens, resp.output_tokens), 6),
                "latency_ms": round(1000 * (time.perf_counter() - t0)), "raw": resp.raw_text}


def _cite(chunk: dict) -> dict:
    return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "title": chunk["title"],
            "section": clean_section(chunk.get("section")), "version": chunk.get("version"), "source": chunk.get("source")}


def render(result: dict) -> str:
    """Людський вигляд: твердження з [n] і список джерел «документ › розділ» лише для процитованих."""
    if result["status"] == "not_found":
        why = f" ({result['missing']})" if result.get("missing") else ""
        return NOT_FOUND_TEXT + why
    by_sid = {s["sid"]: s for s in result["sources"]}
    order: list[str] = []
    lines = []
    for c in result["claims"]:
        nums = []
        for sid in c["sources"]:
            if sid not in order:
                order.append(sid)
            nums.append(order.index(sid) + 1)
        lines.append(f"- {c['text']} " + "".join(f"[{n}]" for n in nums))
    if result["status"] == "partial" and result.get("missing"):
        lines.append(f"\nУ документах немає: {result['missing']}")
    lines.append("\nДжерела:")
    for n, sid in enumerate(order, 1):
        ch = by_sid[sid]["chunk"]
        sec = f" › {ch['section']}" if ch["section"] else ""
        lines.append(f"[{n}] {ch['title']}{sec}" + (f" — {ch['source']}" if ch.get("source") else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc-type", nargs="+")
    ap.add_argument("--stack")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    flt = {}
    if args.doc_type:
        flt["doc_type"] = args.doc_type if len(args.doc_type) > 1 else args.doc_type[0]
    if args.stack:
        flt["stack"] = args.stack
    res = Answerer().answer(args.question, flt or None)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(render(res))
        print(f"\n· {res['status']} · контекст {res['context_tokens']} ток. · ${res['cost_usd']:.5f} · {res['latency_ms']} мс"
              f" · prompt {res['prompt_version']} ({res['prompt_sha']})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
