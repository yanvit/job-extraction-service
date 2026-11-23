"""Оцінка якості пошуку (ДЗ2, крок 2): hit@k (recall@k) і MRR на питаннях з відомими правильними чанками.

    python -m rag.eval_retrieval              # усі конфігурації → evals/results/retrieval.{md,json}
    python -m rag.eval_retrieval --check      # лише перевірити розмітку (кожне питання має правильний чанк)

Конфігурації: чанки (structure | fixed) × пошук (vector | bm25 | hybrid | hybrid_rerank),
плюс hybrid_rerank з фільтром за метаданими.

recall@k тут = частка питань, для яких хоча б один правильний чанк є в топ-k (hit rate).
Для питань, де правильний факт повторюється в кількох документах, цього досить: моделі потрібен один.
MRR = середнє 1/місце першого правильного чанка (0, якщо його немає в топ-10).
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import yaml

from .index import ROOT, load_chunks
from .search import Retriever, matches

QUESTIONS = ROOT / "evals" / "retrieval_questions.yaml"
OUT_DIR = ROOT / "evals" / "results"
KS = (1, 3, 5, 10)
CONFIGS = [  # (назва, чанки, режим, з фільтром)
    ("vector", "structure", "vector", False),
    ("bm25", "structure", "bm25", False),
    ("hybrid", "structure", "hybrid", False),
    ("hybrid_rerank", "structure", "hybrid_rerank", False),
    ("hybrid_rerank + filter", "structure", "hybrid_rerank", True),
    ("vector", "fixed", "vector", False),
    ("hybrid", "fixed", "hybrid", False),
    ("hybrid_rerank", "fixed", "hybrid_rerank", False),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    for a in "’ʼ`‘":                          # різні апострофи в українських текстах
        s = s.replace(a, "'")
    return " ".join(s.split())


def is_relevant(chunk: dict, rules: list[dict]) -> bool:
    text = norm(chunk["text"])
    return any(chunk["doc_id"] == r["doc"] and norm(r["contains"]) in text for r in rules)


def load_questions() -> list[dict]:
    return yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]


def check(questions: list[dict]) -> list[str]:
    """Кожне питання має ≥1 правильний чанк у кожній стратегії, і він проходить свій фільтр."""
    problems = []
    for kind in ("structure", "fixed"):
        chunks = load_chunks(kind)
        for q in questions:
            rel = [c for c in chunks if is_relevant(c, q["relevant"])]
            if not rel:
                problems.append(f"{q['id']} [{kind}]: жодного правильного чанка")
            elif q.get("filter") and not any(matches(c, q["filter"]) for c in rel):
                problems.append(f"{q['id']} [{kind}]: правильні чанки не проходять filter={q['filter']}")
    return problems


def evaluate(questions: list[dict]) -> list[dict]:
    retrievers: dict[str, Retriever] = {}
    rows = []
    for name, kind, mode, use_filter in CONFIGS:
        if kind not in retrievers:
            retrievers[kind] = Retriever(kind)
        r = retrievers[kind]
        r.search("прогрів", mode, k=1)               # завантаження моделей не рахуємо в затримку
        per_q, t0 = [], time.perf_counter()
        for q in questions:
            hits = r.search(q["q"], mode, k=max(KS), flt=q.get("filter") if use_filter else None)
            ranks = [i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, q["relevant"])]
            first = ranks[0] if ranks else None
            per_q.append({"id": q["id"], "first_rank": first,
                          "top1": hits[0].chunk["chunk_id"] if hits else None})
        n = len(questions)
        row = {"config": name, "chunks": kind, "mode": mode, "filter": use_filter,
               "ms_per_query": round(1000 * (time.perf_counter() - t0) / n),
               "mrr": round(sum(1 / p["first_rank"] for p in per_q if p["first_rank"]) / n, 3),
               "per_question": per_q}
        for k in KS:
            row[f"recall@{k}"] = round(sum(1 for p in per_q if p["first_rank"] and p["first_rank"] <= k) / n, 3)
        rows.append(row)
        print(f"{kind:9} {name:24} " + "  ".join(f"R@{k}={row[f'recall@{k}']:.2f}" for k in KS)
              + f"  MRR={row['mrr']:.3f}  {row['ms_per_query']} ms/q")
    return rows


def to_markdown(rows: list[dict], questions: list[dict]) -> str:
    lines = [f"# Якість retrieval ({len(questions)} питань)", "",
             "Згенеровано `make eval-retrieval`. Питання і правила відповідності: `evals/retrieval_questions.yaml`.", "",
             "| Чанки | Конфігурація | recall@1 | recall@3 | recall@5 | recall@10 | MRR | мс/запит |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['chunks']} | {r['config']} | " + " | ".join(f"{r[f'recall@{k}']:.2f}" for k in KS)
                     + f" | {r['mrr']:.3f} | {r['ms_per_query']} |")
    lines += ["", "## Місце першого правильного чанка по питаннях", "",
              "| Питання | " + " | ".join(f"{r['chunks'][:6]} {r['config']}" for r in rows) + " |",
              "|---|" + "---|" * len(rows)]
    for i, q in enumerate(questions):
        cells = [str(r["per_question"][i]["first_rank"] or "—") for r in rows]
        lines.append(f"| {q['id']} {q['q'][:60]} | " + " | ".join(cells) + " |")
    lines += ["", "«—» — правильного чанка немає в топ-10."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    questions = load_questions()
    problems = check(questions)
    for p in problems:
        print("!!", p)
    if problems:
        raise SystemExit(1)
    print(f"розмітка ок: {len(questions)} питань")
    if args.check:
        return
    rows = evaluate(questions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "retrieval.md").write_text(to_markdown(rows, questions), encoding="utf-8")
    print(f"→ {OUT_DIR / 'retrieval.md'}")


if __name__ == "__main__":
    main()
