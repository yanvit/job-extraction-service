"""Scoring-функції для evals RAG (ДЗ2, крок 4).

Детерміновані (без LLM, безкоштовні, однакові на кожному прогоні):
  abstain_ok    — «не знаю» рівно там, де треба: no_answer → not_found; answerable/trap → є відповідь;
  source_hit    — серед процитованих документів є хоча б один з очікуваних;
  key_facts     — частка груп ключових фактів, знайдених у тексті відповіді (цифри нормалізуються);
  must_not_ok   — у відповіді немає типової помилки пастки (напр. літня цифра на питання про зиму).
LLM-суддя (Claude Opus 5.5, промпти prompts/judge/v1):
  judge_correctness — correct / partial / incorrect проти еталонної відповіді (1 / 0.5 / 0);
  judge_faithfulness — для кожного твердження: чи випливає воно з процитованих ним джерел.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / "prompts" / "judge"

# ------------------------------------------------------------------ deterministic


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    for a in "’ʼ`‘":
        s = s.replace(a, "'")
    s = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", s)   # «10 091 049» → «10091049»
    s = re.sub(r"(\d)\s+%", r"\1%", s)                          # «5 %» → «5%»
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                       # «1,5%» → «1.5%»
    return " ".join(s.split())


def contains(text: str, fact: str) -> bool:
    """Ліва межа слова завжди («5%» не знаходиться в «15%»). Права — після цифри («1200» не в «12000»)
    і для коротких назв («go» не в «google»); довші слова ловимо за основою («гарант» → «гарантує»)."""
    t, f = norm(text), norm(fact)
    left = r"(?<!\w)" if re.match(r"\w", f) else ""
    if re.search(r"\d$", f):
        right = r"(?!\d)"
    elif len(f) <= 3:                  # короткі назви («go», «c#») — цілим словом
        right = r"(?!\w)"
    else:
        right = ""
    return re.search(left + re.escape(f) + right, t) is not None


def answer_text(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("claims", []))


def cited_docs(result: dict) -> set[str]:
    cited = {sid for c in result.get("claims", []) for sid in c["sources"]}
    return {s["chunk"]["doc_id"] for s in result.get("sources", []) if s["sid"] in cited}


def deterministic(item: dict, result: dict) -> dict:
    answered = result["status"] != "not_found"
    out = {"abstain_ok": (not answered) if item["category"] == "no_answer" else answered}
    if item["category"] == "no_answer":
        return out
    text = answer_text(result)
    groups = item.get("key_facts") or []
    hit = [any(contains(text, alt) for alt in g) for g in groups]
    out["key_facts"] = round(sum(hit) / len(groups), 3) if groups else None
    out["key_facts_missed"] = [g[0] for g, h in zip(groups, hit) if not h]
    out["source_hit"] = bool(cited_docs(result) & set(item["sources"]))
    if item.get("must_not"):
        bad = [m for m in item["must_not"] if contains(text, m)]
        out["must_not_ok"], out["must_not_found"] = not bad, bad
    return out


# ------------------------------------------------------------------ LLM judge

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
}
FAITH_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "supported": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["index", "supported", "reason"]}}},
    "required": ["claims"],
}
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def load_judge(version: str = "v1") -> dict:
    d = JUDGE_DIR / version
    p = {"version": version, "correctness": (d / "correctness.md").read_text(encoding="utf-8").strip(),
         "faithfulness": (d / "faithfulness.md").read_text(encoding="utf-8").strip()}
    p["sha"] = hashlib.sha256((p["correctness"] + p["faithfulness"]).encode()).hexdigest()[:12]
    return p


def render_answer(result: dict) -> str:
    if result["status"] == "not_found":
        return "«У документах цього немає.» " + (result.get("missing") or "")
    lines = [f"- {c['text']}" for c in result["claims"]]
    if result["status"] == "partial" and result.get("missing"):
        lines.append(f"(Чого немає в документах: {result['missing']})")
    return "\n".join(lines)


def judge_correctness(client, judge: dict, question: str, expected: str, answer: str):
    from jobex.clients.base import Sampling
    user = (f"<question>\n{question}\n</question>\n\n<reference_answer>\n{expected}\n</reference_answer>\n\n"
            f"<system_answer>\n{answer}\n</system_answer>")
    resp = client.extract(judge["correctness"], user, CORRECTNESS_SCHEMA, Sampling("api_default"))
    d = resp.data or {}
    verdict = d.get("verdict") if d.get("verdict") in VERDICT_SCORE else "incorrect"
    return {"verdict": verdict, "score": VERDICT_SCORE[verdict], "reason": d.get("reason", "parse error")}, resp


def judge_faithfulness(client, judge: dict, result: dict, blocks: dict[str, str]):
    """Кожне твердження — лише з тими джерелами, які воно саме цитує."""
    from jobex.clients.base import Sampling
    parts = []
    for i, c in enumerate(result["claims"], 1):
        src = "\n\n".join(blocks[s] for s in c["sources"] if s in blocks)
        parts.append(f"<claim index=\"{i}\">\n{c['text']}\n</claim>\n<cited_sources index=\"{i}\">\n{src}\n</cited_sources>")
    resp = client.extract(judge["faithfulness"], "\n\n".join(parts), FAITH_SCHEMA, Sampling("api_default"))
    rows = {r.get("index"): r for r in (resp.data or {}).get("claims", [])}
    per = [{"text": c["text"], "supported": bool(rows.get(i, {}).get("supported", False)),
            "reason": rows.get(i, {}).get("reason", "немає оцінки")} for i, c in enumerate(result["claims"], 1)]
    return per, resp
