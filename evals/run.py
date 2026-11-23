"""Прогін evals RAG: тестсет → відповіді → scoring → метрики → пороги (exit 1, якщо нижче порогу).

    python -m evals.run                 # усе: відповіді, детерміновані перевірки, LLM-суддя, faithfulness
    python -m evals.run --no-judge      # лише детерміновані метрики (без витрат на суддю)
    python -m evals.run --offline       # лише з кешу (evals/cache); промах кешу — помилка, API не викликається

Результати: evals/results/answers.jsonl (по рядку на питання), evals/results/metrics.{json,md}.
Відповіді моделей кешуються в evals/cache/ (у git), тож повторний прогін без змін безкоштовний.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from rag.answer import ACFG, Answerer, build_client, price
from rag.llm_cache import CachedClient

from .scoring import (deterministic, judge_correctness, judge_faithfulness, load_judge, render_answer)

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
CFG = yaml.safe_load((EVALS / "config.yaml").read_text(encoding="utf-8"))
RESULTS = EVALS / "results"


def load_testset() -> list[dict]:
    return yaml.safe_load((EVALS / "testset.yaml").read_text(encoding="utf-8"))["questions"]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def evaluate_one(item: dict, answerer: Answerer, judge_client, judge: dict | None) -> dict:
    res = answerer.answer(item["q"], item.get("filter"))
    blocks = res.pop("_blocks", {})
    row = {"id": item["id"], "category": item["category"], "q": item["q"], "status": res["status"],
           "gate": res.get("gate"), "claims": res["claims"], "missing": res.get("missing"),
           "cited_docs": sorted({s["chunk"]["doc_id"] for s in res["sources"]
                                 if s["sid"] in {x for c in res["claims"] for x in c["sources"]}}),
           "top_score": res["top_score"], "context_tokens": res["context_tokens"],
           "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"], "cost_usd": res["cost_usd"],
           "latency_ms": res["latency_ms"], "dropped_claims": len(res.get("dropped_claims", [])),
           "prompt_version": res["prompt_version"], "prompt_sha": res["prompt_sha"]}
    row.update(deterministic(item, res))
    row["judge_cost_usd"] = 0.0
    if judge and item["category"] != "no_answer":
        if res["status"] == "not_found":            # безпідставна відмова — детерміновано incorrect, суддю не кличемо
            row["judge"] = {"verdict": "incorrect", "score": 0.0, "reason": "система відмовилась відповідати"}
        else:
            row["judge"], r1 = judge_correctness(judge_client, judge, item["q"], item["expected"], render_answer(res))
            row["faithfulness"], r2 = judge_faithfulness(judge_client, judge, res, blocks)
            row["judge_cost_usd"] = round(price(judge_client.model_id, r1.input_tokens + r2.input_tokens,
                                                r1.output_tokens + r2.output_tokens), 6)
        row["judge_version"], row["judge_sha"] = judge["version"], judge["sha"]
    return row


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["category"] != "no_answer"]
    noans = [r for r in rows if r["category"] == "no_answer"]
    claims = [c for r in ans for c in r.get("faithfulness", [])]
    llm_rows = [r for r in rows if r["input_tokens"]]
    m = {
        "n_questions": len(rows),
        "by_category": {c: sum(r["category"] == c for r in rows) for c in ("answerable", "trap", "no_answer")},
        "abstain_accuracy": mean([1.0 if r["abstain_ok"] else 0.0 for r in rows]),
        "false_refusal_rate": mean([0.0 if r["abstain_ok"] else 1.0 for r in ans]),
        "hallucination_rate_no_answer": mean([0.0 if r["abstain_ok"] else 1.0 for r in noans]),
        "source_hit": mean([1.0 if r["source_hit"] else 0.0 for r in ans]),
        "key_facts": mean([r["key_facts"] for r in ans]),
        "must_not_ok": mean([1.0 if r["must_not_ok"] else 0.0 for r in ans if "must_not_ok" in r]),
        "judge_correct": mean([r["judge"]["score"] for r in ans if "judge" in r]),
        "judge_correct_trap": mean([r["judge"]["score"] for r in ans if "judge" in r and r["category"] == "trap"]),
        "faithfulness_claims": mean([1.0 if c["supported"] else 0.0 for c in claims]),
        "faithful_answers": mean([1.0 if all(c["supported"] for c in r["faithfulness"]) else 0.0
                                  for r in ans if r.get("faithfulness")]),
        "n_claims": len(claims),
        "refused_by_gate": sum(r["gate"] == "retrieval" for r in rows),
        "refused_by_model": sum(r["gate"] == "model" for r in rows),
        "errors": sum(r["gate"] == "error" for r in rows),
        "cost_per_1000_questions_usd": round(1000 * sum(r["cost_usd"] for r in rows) / len(rows), 2),
        "judge_cost_run_usd": round(sum(r["judge_cost_usd"] for r in rows), 4),
        "latency_p50_ms": round(statistics.median([r["latency_ms"] for r in llm_rows])) if llm_rows else 0,
        "latency_p95_ms": round(pct([r["latency_ms"] for r in llm_rows], 0.95)) if llm_rows else 0,
        "avg_context_tokens": round(statistics.mean([r["context_tokens"] for r in llm_rows])) if llm_rows else 0,
    }
    return m


def check_thresholds(m: dict) -> list[str]:
    fails = []
    for key, th in CFG["thresholds"].items():
        val = m.get(key)
        if key == "retrieval_recall@5":
            val = retrieval_recall5()
            m[key] = val
        if val is None:
            continue
        bad = val > th["max"] if "max" in th else val < th["min"]
        if bad:
            fails.append(f"{key} = {val} ({'max' if 'max' in th else 'min'} {th.get('max', th.get('min'))})")
    return fails


def retrieval_recall5() -> float | None:
    p = RESULTS / "retrieval.json"
    if not p.exists():
        return None
    for r in json.loads(p.read_text(encoding="utf-8")):
        if r["chunks"] == "structure" and r["config"] == ACFG["retrieve_mode"]:
            return r["recall@5"]
    return None


def to_markdown(m: dict, rows: list[dict], fails: list[str], cache_info: str) -> str:
    th = CFG["thresholds"]
    lines = [f"# Evals RAG — {m['n_questions']} питань", "",
             f"Згенеровано `make evals`. {cache_info}", "",
             "| Метрика | Значення | Поріг |", "|---|---|---|"]
    names = {"abstain_accuracy": "«не знаю» там, де треба (усі питання)",
             "hallucination_rate_no_answer": "відповів на питання без відповіді",
             "false_refusal_rate": "відмовився, хоча відповідь є",
             "source_hit": "правильне джерело в цитатах", "key_facts": "ключові факти у відповіді",
             "must_not_ok": "без типової помилки пастки", "judge_correct": "LLM-суддя: правильність",
             "judge_correct_trap": "LLM-суддя: правильність на пастках",
             "faithfulness_claims": "faithfulness: підтверджені твердження",
             "faithful_answers": "faithfulness: відповіді без жодного непідтвердженого",
             "retrieval_recall@5": "retrieval recall@5 (hybrid+rerank)"}
    for k, label in names.items():
        if m.get(k) is None:
            continue
        t = th.get(k, {})
        lines.append(f"| {label} | {m[k]} | {('≤ ' + str(t['max'])) if 'max' in t else ('≥ ' + str(t['min'])) if 'min' in t else ''} |")
    lines += ["", f"Вартість відповідей: **${m['cost_per_1000_questions_usd']} на 1000 питань** "
              f"(відмови на порозі reranker-а безкоштовні: {m['refused_by_gate']} з {m['n_questions']}). "
              f"Затримка p50 {m['latency_p50_ms']} мс, p95 {m['latency_p95_ms']} мс. "
              f"Суддя за цей прогін: ${m['judge_cost_run_usd']}.", "",
              "**Статус: " + ("❌ нижче порогу: " + "; ".join(fails) if fails else "✅ усі пороги пройдено") + "**", "",
              "## По питаннях", "",
              "| id | тип | статус | джерело | факти | суддя | faithfulness | цитовані документи |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fa = r.get("faithfulness")
        lines.append(" | ".join([
            f"| {r['id']}", r["category"], r["status"] + (f" ({r['gate']})" if r.get("gate") else ""),
            "—" if r["category"] == "no_answer" else ("✅" if r["source_hit"] else "❌"),
            "—" if r.get("key_facts") is None else f"{r['key_facts']:.2f}",
            r["judge"]["verdict"] if "judge" in r else "—",
            f"{sum(c['supported'] for c in fa)}/{len(fa)}" if fa else "—",
            ", ".join(r["cited_docs"]) or "—"]) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--offline", action="store_true", help="лише кеш; промах — помилка")
    ap.add_argument("--only", nargs="+", help="лише ці id")
    args = ap.parse_args()

    items = [q for q in load_testset() if not args.only or q["id"] in args.only]
    cache = EVALS / "cache"
    ans_client = CachedClient(build_client(ACFG["model"], ACFG["max_tokens"]) if not args.offline else _Offline(ACFG["model"]),
                              cache / "answers", read_only=args.offline)
    answerer = Answerer(client=ans_client)
    judge = None if args.no_judge else load_judge(CFG["judge"]["prompt"])
    judge_client = None
    if judge:
        jid = CFG["judge"]["model"]
        judge_client = CachedClient(build_client(jid) if not args.offline else _Offline(jid),
                                    cache / "judge", read_only=args.offline)
        judge_client.model_id = jid

    _ = answerer.retriever                                     # завантажити моделі пошуку до потоків
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CFG["concurrency"]) as ex:
        rows = list(ex.map(lambda it: evaluate_one(it, answerer, judge_client, judge), items))
    m = aggregate(rows)
    fails = check_thresholds(m) if not args.only else []
    hits = ans_client.hits + (judge_client.hits if judge_client else 0)
    misses = ans_client.misses + (judge_client.misses if judge_client else 0)
    cache_info = f"Кеш LLM: {hits} з кешу, {misses} нових викликів; {time.perf_counter() - t0:.0f} с."

    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.only:
        with (RESULTS / "answers.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (RESULTS / "metrics.json").write_text(json.dumps({**m, "failed_thresholds": fails}, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        (RESULTS / "metrics.md").write_text(to_markdown(m, rows, fails, cache_info), encoding="utf-8")
    for k, v in m.items():
        print(f"{k:32} {v}")
    print(cache_info)
    if args.only:
        for r in rows:
            print(json.dumps({k: r.get(k) for k in ("id", "status", "gate", "claims", "missing", "judge", "faithfulness",
                                                     "key_facts_missed", "source_hit", "cited_docs")},
                             ensure_ascii=False, indent=1))
    if fails:
        print("\nFAILED thresholds:\n  " + "\n  ".join(fails))
        return 1
    print("\nall thresholds passed" if not args.only else "")
    return 0


class _Offline:
    """Заглушка клієнта для --offline: кеш віддає все, а сюди звертатись не можна."""

    def __init__(self, model_id: str):
        from rag.answer import _model_cfg
        self.model = _model_cfg(model_id)[0]["model"]

    def extract(self, *a, **kw):
        raise RuntimeError("offline mode: no API calls")


if __name__ == "__main__":
    sys.exit(main())
