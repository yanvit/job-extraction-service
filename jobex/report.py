"""Генерація таблиць для REPORT.md і файлу розбіжностей з golden set (для розбору помилок)."""
from __future__ import annotations

import json
from pathlib import Path

from .evaluate import load_golden, load_run, norm_loc, score_doc, skill_keys
from .schema import FIELDS

START, END = "<!-- AUTO:METRICS:START -->", "<!-- AUTO:METRICS:END -->"

SHORT = {"title": "title", "company": "comp", "role_category": "role", "seniority": "senior",
         "work_format": "format", "employment_type": "empl", "locations": "loc", "salary": "salary",
         "experience_years_min": "exp", "skills": "skills", "languages": "lang", "posting_language": "p_lang"}


def _f(v, fmt="{:.2f}", none="—"):
    return none if v is None else fmt.format(v)


def metrics_tables(metrics: list[dict]) -> str:
    rows = sorted(metrics, key=lambda m: (m["model_id"], m["prompt_version"], m["sampling"]))
    out = ["### Зведена таблиця: модель × версія промпту × семплінг", "",
           "| Модель | Промпт | Семплінг (що реально пішло в API) | Валідні, % (1-а спроба / після repair) | "
           "Точність по полях, golden | p50 / p95, с | $ / 1000 док | Токени in / out (сер.) |",
           "|---|---|---|---|---|---|---|---|"]
    for m in rows:
        sent = json.loads(m["sampling_sent"] or "{}")
        sent.pop("structured_mode", None)
        sent_s = ", ".join(f"{k}={v}" for k, v in sent.items()) or "API default"
        out.append(
            f'| {m["model_id"]} | {m["prompt_version"]} | {m["sampling"]} ({sent_s}) | '
            f'{m["valid_first_try_pct"]:.0f} / {m["valid_pct"]:.0f} | '
            f'**{_f(m["field_acc_overall"], "{:.3f}")}** (n={m["n_golden"]}) | '
            f'{_f(m["latency_p50_ms"] and m["latency_p50_ms"] / 1000, "{:.1f}")} / '
            f'{_f(m["latency_p95_ms"] and m["latency_p95_ms"] / 1000, "{:.1f}")} | '
            f'{_f(m["cost_per_1000_docs_usd"], "{:.3f}")} | '
            f'{m["avg_input_tokens"]:.0f} / {m["avg_output_tokens"]:.0f} |')
    out += ["", "### Точність по кожному полю (golden set)", "",
            "| Прогін | " + " | ".join(SHORT[f] for f in FIELDS) + " | skills.required |",
            "|---|" + "---|" * (len(FIELDS) + 1)]
    for m in rows:
        out.append(f'| {m["run"]} | ' + " | ".join(_f(m[f"acc_{f}"]) for f in FIELDS)
                   + f' | {_f(m["skills_required_acc"])} |')
    out += ["", "_Валідні — % документів прогону, що пройшли Pydantic-валідацію. "
            "Точність — середнє по 12 полях на golden set; невалідна відповідь = 0 по всіх полях. "
            "Списки (locations, skills, languages) — F1; salary — частка збігів з {min, max, currency, period}; "
            "title/company — нечіткий збіг. Latency — end-to-end на документ, включно з ретраями та repair. "
            "Вартість — за прайсом з config.yaml; для локальної моделі API-вартість 0._"]
    return "\n".join(out)


def update_report(report_path: Path, metrics: list[dict]) -> None:
    report_path = Path(report_path)
    block = f"{START}\n{metrics_tables(metrics)}\n{END}"
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        _, post = rest.split(END, 1)
        text = pre + block + post
    else:
        text = text + "\n\n" + block + "\n"
    report_path.write_text(text, encoding="utf-8")


def _diff(field: str, exp, got) -> str:
    """Для списків показуємо лише різницю (−пропущено / +зайве), для решти — очікувалось → отримали."""
    if field in ("skills", "locations", "languages"):
        key = {"skills": lambda x: norm_skill(x["name"]), "locations": norm_loc,
               "languages": lambda x: f'{x["language"]}:{x["level"]}'}[field]
        show = {"skills": lambda x: x["name"], "locations": lambda x: x,
                "languages": lambda x: f'{x["language"]}:{x["level"]}'}[field]
        if field == "skills":  # складені назви ('BAS/1С') розкладаємо так само, як у метриці
            e = {k: x["name"] for x in exp or [] for k in skill_keys(x["name"])}
            o = {k: x["name"] for x in got or [] for k in skill_keys(x["name"])}
        else:
            e = {key(x): show(x) for x in exp or []}
            o = {key(x): show(x) for x in got or []}
        miss = [e[k] for k in e if k not in o]
        extra = [o[k] for k in o if k not in e]
        parts = ([f"−пропущено: {', '.join(miss)}"] if miss else []) + ([f"+зайве: {', '.join(extra)}"] if extra else [])
        return "; ".join(parts) or "той самий набір, різниться required"
    return f"{json.dumps(exp, ensure_ascii=False)} → {json.dumps(got, ensure_ascii=False)}"


def write_golden_diff(results_dir: Path, golden_path: Path) -> Path:
    """results/golden_diff.md: для кожного прогону — поля, де модель розійшлась з golden."""
    results_dir = Path(results_dir)
    golden = load_golden(golden_path)
    lines = ["# Розбіжності з golden set", "",
             "Згенеровано `python main.py report`. Формат: поле (score): очікувалось → отримали.", ""]
    for p in sorted((results_dir / "runs").glob("*.jsonl")):
        recs = load_run(p)
        lines.append(f"## {p.stem}")
        for r in recs:
            if r["doc_id"] not in golden:
                continue
            g = golden[r["doc_id"]]
            if r["output"] is None:
                lines.append(f"- **{r['doc_id']}**: FAILED — {r.get('error') or r.get('validation_errors')}")
                continue
            sc = score_doc(r["output"], g)
            bad = [f for f in FIELDS if sc[f] < 1]
            if not bad:
                continue
            lines.append(f"- **{r['doc_id']}** ({r['status']})")
            for f in bad:
                lines.append(f"  - `{f}` ({sc[f]:.2f}): {_diff(f, g.get(f), r['output'].get(f))}")
        lines.append("")
    out = results_dir / "golden_diff.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
