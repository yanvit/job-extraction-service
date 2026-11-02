#!/usr/bin/env python3
"""data/corpus/djinni_se_raw.jsonl → data/corpus/vacancies/<id>.md (по файлу на вакансію).

Markdown-формат корпусу: заголовок, метадані (frontmatter), опис з виділеними підзаголовками і списками,
сайдбар Djinni (рядок з « | ») — окремим розділом «Метадані Djinni».
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "corpus" / "djinni_se_raw.jsonl"
OUT = ROOT / "data" / "corpus" / "vacancies"

HEADING_RE = re.compile(r"^[^\n]{2,60}:$")      # «Requirements:», «Що ми пропонуємо:»


def to_markdown(row: dict) -> str:
    lines = [l.rstrip() for l in row["text"].splitlines()]
    sidebar = [l for l in lines if l.count(" | ") >= 3]
    body = [l for l in lines if l not in sidebar]
    if body and body[0].strip() == row["title"].strip():
        body = body[1:]
    md = ["---", f"id: djinni-{row['id']}", f"title: {json.dumps(row['title'], ensure_ascii=False)}",
          f"source: {row['url']}", f"stack: {row['keyword']}", f"published: {row['published'] or 'unknown'}",
          "---", "", f"# {row['title']}", ""]
    for l in body:
        s = l.strip()
        if not s:
            md.append("")
        elif s.startswith(("- ", "• ", "* ")):
            md.append("- " + s[2:].strip())
        elif HEADING_RE.match(s) and not s.startswith(("http", "-")):
            md += ["", f"## {s.rstrip(':')}", ""]
        else:
            md.append(s)
    if sidebar:
        md += ["", "## Метадані Djinni", ""] + [f"- {p.strip()}" for l in sidebar for p in l.split(" | ") if p.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(md)).strip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        (OUT / f"djinni-{r['id']}.md").write_text(to_markdown(r), encoding="utf-8")
    print(f"{len(rows)} vacancies → {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
