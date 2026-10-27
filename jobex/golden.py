"""Інструменти для golden set.

    python -m jobex.golden check    # кожен запис валідний за схемою JobPosting, doc_id існує
    python -m jobex.golden review   # data/golden_review.md: текст вакансії поруч із розміткою для звірки
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .schema import JobPosting

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
RAW = ROOT / "data" / "raw"


def load() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def check() -> int:
    rows, bad = load(), 0
    ids = [r["doc_id"] for r in rows]
    if len(ids) != len(set(ids)):
        print("duplicate doc_id in golden.jsonl"); bad += 1
    for r in rows:
        if not (RAW / f"{r['doc_id']}.txt").exists():
            print(f"{r['doc_id']}: raw text not found"); bad += 1
        try:
            JobPosting.model_validate(r["expected"])
        except ValidationError as e:
            print(f"{r['doc_id']}: {e}"); bad += 1
    print(f"{len(rows)} golden records, {bad} problems")
    return 1 if bad else 0


def review() -> int:
    out = ["# Golden set: звірка", "",
           "Для кожної вакансії: оригінальний текст і розмітка. Звір кожне поле з текстом; правила — `data/LABELING.md`.", ""]
    for r in load():
        e = r["expected"]
        text = (RAW / f"{r['doc_id']}.txt").read_text(encoding="utf-8")
        sal = e["salary"]
        sal_s = "null" if sal is None else f'{sal["min"]} – {sal["max"]} {sal["currency"]} / {sal["period"]}'
        skills = ", ".join(f'{s["name"]}{"" if s["required"] else " (+)"}' for s in e["skills"]) or "—"
        langs = ", ".join(f'{l["language"]}:{l["level"]}' for l in e["languages"]) or "—"
        out += [f"## {r['doc_id']}", "", "| Поле | Розмітка |", "|---|---|",
                f'| title | {e["title"]} |', f'| company | {e["company"]} |',
                f'| role_category | {e["role_category"]} |', f'| seniority | {e["seniority"]} |',
                f'| work_format | {e["work_format"]} |', f'| employment_type | {e["employment_type"]} |',
                f'| locations | {", ".join(e["locations"]) or "—"} |', f"| salary | {sal_s} |",
                f'| experience_years_min | {e["experience_years_min"]} |',
                f"| skills ((+) = буде плюсом) | {skills} |", f"| languages | {langs} |",
                f'| posting_language | {e["posting_language"]} |', "",
                "<details><summary>Текст вакансії</summary>", "", "```text", text.strip(), "```", "", "</details>", ""]
    path = ROOT / "data" / "golden_review.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"written {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    sys.exit({"check": check, "review": review}[cmd]())
