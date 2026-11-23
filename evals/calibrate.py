"""Калібрування LLM-судді на ручній розмітці.

    python -m evals.calibrate export   # evals/calibration/sheet.yaml: відповіді для ручної оцінки (без вердиктів судді)
    # → вписати human: correct | partial | incorrect у кожен пункт (і за бажання note)
    python -m evals.calibrate score    # суддя оцінює ті самі відповіді → згода з людиною → evals/results/calibration.md

У лист потрапляють реальні відповіді системи з останнього прогону + кілька навмисно зіпсованих
(підмінена цифра чи назва). Без них усі відповіді були б правильні, і ми б не знали, чи суддя
взагалі здатен сказати «incorrect».
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

from rag.answer import build_client
from rag.llm_cache import CachedClient

from .scoring import judge_correctness, load_judge

EVALS = Path(__file__).resolve().parent
SHEET = EVALS / "calibration" / "sheet.yaml"
KEY = EVALS / "calibration" / "key.json"           # c01 → t07~perturbed; не відкривати до ручної оцінки
CFG = yaml.safe_load((EVALS / "config.yaml").read_text(encoding="utf-8"))
N_REAL = 12
# Зіпсовані відповіді: (id, що замінити, на що). Беремо ті, де заміна точно знайдеться в тексті відповіді.
PERTURB = [("t07", "10 091 049", "8 285 700"), ("t07", "10091049", "8285700"),
           ("t09", "3900", "3400"), ("t09", "3 900", "3 400"),
           ("t15", "4500", "4400"), ("t15", "4 500", "4 400"),
           ("t12", "Lead Rust Engineer", "Senior Golang Developer"),
           ("t04", "12 ", "6 "), ("t18", "5%", "1,5%"), ("t18", "5 %", "1,5 %")]


def answer_text(row: dict) -> str:
    return "\n".join(f"- {c['text']}" for c in row["claims"])


def export() -> None:
    rows = [json.loads(l) for l in (EVALS / "results" / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    tests = {q["id"]: q for q in yaml.safe_load((EVALS / "testset.yaml").read_text(encoding="utf-8"))["questions"]}
    answered = [r for r in rows if r["category"] != "no_answer" and r["status"] != "not_found"]
    random.Random(42).shuffle(answered)
    items = [{"cid": r["id"], "q": r["q"], "expected": tests[r["id"]]["expected"], "answer": answer_text(r),
              "human": None, "note": ""} for r in answered[:N_REAL]]
    done = set()
    for rid, old, new in PERTURB:
        r = next((x for x in rows if x["id"] == rid), None)
        if not r or rid in done or old not in answer_text(r):
            continue
        done.add(rid)
        items.append({"cid": f"{rid}~perturbed", "q": r["q"], "expected": tests[rid]["expected"],
                      "answer": answer_text(r).replace(old, new), "human": None, "note": ""})
    random.Random(7).shuffle(items)                       # щоб зіпсовані не йшли підряд
    key = {}
    for n, it in enumerate(items, 1):                     # нейтральні номери: з id не видно, що зіпсовано
        key[f"c{n:02d}"] = it["cid"]
        it["cid"] = f"c{n:02d}"
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    header = ("# Ручна оцінка відповідей для калібрування LLM-судді.\n"
              "# Для кожного пункту впишіть human: correct | partial | incorrect (порівнюючи answer з expected),\n"
              "# за потреби — note. Критерії ті самі, що для судді: prompts/judge/v1/correctness.md.\n"
              "# Частину відповідей навмисно зіпсовано — оцінюйте кожну чесно. Не відкривайте key.json до кінця оцінки.\n"
              "# Потім: make calibrate\n")
    SHEET.write_text(header + yaml.safe_dump({"items": items}, allow_unicode=True, sort_keys=False, width=120),
                     encoding="utf-8")
    print(f"→ {SHEET} ({len(items)} пунктів, з них {len(done)} зіпсованих)")


def kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def score() -> int:
    items = yaml.safe_load(SHEET.read_text(encoding="utf-8"))["items"]
    todo = [i["cid"] for i in items if i.get("human") not in ("correct", "partial", "incorrect")]
    if todo:
        print(f"ще не оцінено вручну: {len(todo)} з {len(items)} ({', '.join(todo[:5])}…). Заповніть {SHEET}")
        return 0
    judge = load_judge(CFG["judge"]["prompt"])
    client = CachedClient(build_client(CFG["judge"]["model"]), EVALS / "cache" / "judge")
    rows = []
    for it in items:
        v, _ = judge_correctness(client, judge, it["q"], it["expected"], it["answer"])
        rows.append({**it, "judge": v["verdict"], "judge_reason": v["reason"]})
    key = json.loads(KEY.read_text(encoding="utf-8"))
    for r in rows:
        r["cid"] = f"{r['cid']} ({key.get(r['cid'], '?')})"
    h, j = [r["human"] for r in rows], [r["judge"] for r in rows]
    hb = ["correct" if x == "correct" else "not" for x in h]
    jb = ["correct" if x == "correct" else "not" for x in j]
    exact = sum(x == y for x, y in zip(h, j)) / len(rows)
    binary = sum(x == y for x, y in zip(hb, jb)) / len(rows)
    pert = [r for r in rows if "~perturbed" in r["cid"]]
    lines = [f"# Калібрування LLM-судді ({len(rows)} відповідей, оцінених вручну)", "",
             f"- Суддя: `{CFG['judge']['model']}`, промпт `{judge['version']}` ({judge['sha']}).",
             f"- Точна згода (correct / partial / incorrect): **{exact:.0%}**, Cohen's κ = {kappa(h, j)}.",
             f"- Згода «correct vs не correct»: **{binary:.0%}**, κ = {kappa(hb, jb)}.",
             f"- Зіпсовані відповіді, які суддя не назвав correct: {sum(r['judge'] != 'correct' for r in pert)} з {len(pert)}.",
             "", "| id | людина | суддя | пояснення судді |", "|---|---|---|---|"]
    for r in rows:
        mark = "" if r["human"] == r["judge"] else " ⚠️"
        lines.append(f"| {r['cid']} | {r['human']} | {r['judge']}{mark} | {r['judge_reason'][:160]} |")
    out = EVALS / "results" / "calibration.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    sys.exit(export() or 0 if cmd == "export" else score())
