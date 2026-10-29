"""Метрики: точність по полях на golden set, % валідних, latency p50/p95, вартість на 1000 документів."""
from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

from .schema import FIELDS

# ---------------------------------------------------------------- normalizers

_SKILL_ALIASES = {
    "postgres": "postgresql", "k8s": "kubernetes", "js": "javascript", "ts": "typescript",
    "golang": "go", "python3": "python", "1с": "1c", "reactjs": "react", "nodejs": "node", "gcp": "googlecloud",
    "googlecloudplatform": "googlecloud", "amazonwebservices": "aws", "csharp": "c#",
    "1centerprise83": "1c", "1centerprise": "1c", "1cпідприємство83": "1c", "1cпідприємство": "1c",
    "1cпідприємство8": "1c", "1c83": "1c", "basкуп": "baskup", "bas1c": "bas", "basbaf": "bas", "1cbas": "bas",
}


def norm_text(s: Optional[str]) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"[\"'«»“”’`]", "", s)
    return re.sub(r"\s+", " ", s)


def norm_skill(s: str) -> str:
    s = norm_text(s)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("1с", "1c")      # кирилична "С" в "1С"
    s = re.sub(r"[\s.\-_/:]", "", s)
    return _SKILL_ALIASES.get(s, s)


_LOC_ALIASES = {"kiev": "kyiv", "київ": "kyiv", "львів": "lviv", "україна": "ukraine", "ua": "ukraine",
                "весь світ": "worldwide", "remote": "worldwide", "європа": "europe", "єс": "eu",
                "european union": "eu", "eu countries": "eu", "united states": "usa", "us": "usa",
                "krakow": "kraków", "kyiv region": "kyiv"}


def norm_loc(s: str) -> str:
    s = norm_text(s)
    return _LOC_ALIASES.get(s, s)


# ---------------------------------------------------------------- scorers (0..1)

def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def score_exact(p, g) -> float:
    return float(p == g)


def score_title(p, g) -> float:
    a, b = norm_text(p), norm_text(g)
    return float(a == b or SequenceMatcher(None, a, b).ratio() >= 0.85)


def score_company(p, g) -> float:
    if p is None or g is None:
        return float(p is None and g is None)
    a, b = norm_text(p), norm_text(g)
    return float(a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.8)


def score_locations(p, g) -> float:
    return _f1({norm_loc(x) for x in p or []}, {norm_loc(x) for x in g or []})


def _num_eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.01)


def score_salary(p, g) -> float:
    if p is None or g is None:
        return float(p is None and g is None)
    parts = [_num_eq(p.get("min"), g.get("min")), _num_eq(p.get("max"), g.get("max")),
             p.get("currency") == g.get("currency"), p.get("period") == g.get("period")]
    return sum(parts) / len(parts)


def score_years(p, g) -> float:
    return float(_num_eq(p, g))


_NO_SPLIT = {"ci/cd", "tcp/ip", "a/b", "ui/ux", "i/o"}


def skill_keys(name: str) -> set[str]:
    """'BAS/1С' -> {bas, 1c}: складені назви через '/' рахуємо як кілька навичок (крім CI/CD тощо)."""
    if "/" in name and norm_text(name) not in _NO_SPLIT:
        return {norm_skill(part) for part in name.split("/") if part.strip()}
    return {norm_skill(name)}


def _skill_set(skills) -> set[str]:
    return set().union(*[skill_keys(s["name"]) for s in skills or []]) if skills else set()


def score_skills(p, g) -> float:
    return _f1(_skill_set(p), _skill_set(g))


def skills_required_accuracy(p, g) -> Optional[float]:
    """Діагностика: серед знайдених навичок — чи правильно визначено required."""
    gp = {k: s["required"] for s in g or [] for k in skill_keys(s["name"])}
    pp = {k: s["required"] for s in p or [] for k in skill_keys(s["name"])}
    common = gp.keys() & pp.keys()
    if not common:
        return None
    return sum(gp[k] == pp[k] for k in common) / len(common)


def score_languages(p, g) -> float:
    return _f1({(x["language"].lower(), x["level"]) for x in p or []},
               {(x["language"].lower(), x["level"]) for x in g or []})


SCORERS: dict[str, Callable] = {
    "title": score_title, "company": score_company, "role_category": score_exact,
    "seniority": score_exact, "work_format": score_exact, "employment_type": score_exact,
    "locations": score_locations, "salary": score_salary, "experience_years_min": score_years,
    "skills": score_skills, "languages": score_languages, "posting_language": score_exact,
}
assert set(SCORERS) == set(FIELDS), "every schema field must have a scorer"


def score_doc(pred: Optional[dict], gold: dict) -> dict[str, float]:
    if pred is None:  # невалідна відповідь = 0 по всіх полях
        return {f: 0.0 for f in FIELDS}
    return {f: SCORERS[f](pred.get(f), gold.get(f)) for f in FIELDS}


# ---------------------------------------------------------------- aggregation

def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load_golden(path: Path) -> dict[str, dict]:
    gold = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["doc_id"]] = row["expected"]
    return gold


def load_run(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate_run(records: list[dict], golden: dict[str, dict]) -> tuple[dict, list[dict]]:
    first = records[0]
    n = len(records)
    status = [r["status"] for r in records]
    lat = [r["latency_ms"] for r in records if r["status"] != "failed" or r.get("validation_errors")]
    per_doc = []
    for r in records:
        if r["doc_id"] in golden:
            sc = score_doc(r["output"], golden[r["doc_id"]])
            req = skills_required_accuracy((r["output"] or {}).get("skills"), golden[r["doc_id"]].get("skills"))
            per_doc.append({"doc_id": r["doc_id"], **sc, "skills_required_acc": req})
    field_acc = {f: (sum(d[f] for d in per_doc) / len(per_doc) if per_doc else None) for f in FIELDS}
    req_vals = [d["skills_required_acc"] for d in per_doc if d["skills_required_acc"] is not None]
    costs = [r["cost_usd"] for r in records]
    m = {
        "run": f'{first["model_id"]}__{first["prompt_version"]}__{first["sampling"]["name"]}',
        "model_id": first["model_id"], "prompt_version": first["prompt_version"],
        "sampling": first["sampling"]["name"],
        "sampling_sent": json.dumps(next((r["sampling_sent"] for r in records if r.get("sampling_sent") is not None), {})),
        "n_docs": n, "n_golden": len(per_doc),
        "valid_pct": 100 * sum(s in ("ok", "repaired") for s in status) / n,
        "valid_first_try_pct": 100 * status.count("ok") / n,
        "repaired_pct": 100 * status.count("repaired") / n,
        "failed_pct": 100 * status.count("failed") / n,
        "field_acc_overall": (sum(field_acc.values()) / len(field_acc)) if per_doc else None,
        **{f"acc_{f}": v for f, v in field_acc.items()},
        "skills_required_acc": (sum(req_vals) / len(req_vals)) if req_vals else None,
        "latency_p50_ms": percentile(lat, 0.50), "latency_p95_ms": percentile(lat, 0.95),
        "avg_input_tokens": sum(r["input_tokens"] for r in records) / n,
        "avg_output_tokens": sum(r["output_tokens"] for r in records) / n,
        "cost_per_1000_docs_usd": 1000 * sum(costs) / n,
    }
    return m, per_doc


def evaluate_all(results_dir: Path, golden_path: Path) -> list[dict]:
    results_dir = Path(results_dir)
    golden = load_golden(golden_path)
    metrics, per_doc_rows = [], []
    for p in sorted((results_dir / "runs").glob("*.jsonl")):
        recs = load_run(p)
        if not recs:
            continue
        m, per_doc = evaluate_run(recs, golden)
        metrics.append(m)
        per_doc_rows += [{"run": m["run"], **d} for d in per_doc]
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if metrics:
        with (results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(metrics[0].keys()))
            w.writeheader()
            w.writerows(metrics)
    if per_doc_rows:
        with (results_dir / "golden_scores.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_doc_rows[0].keys()))
            w.writeheader()
            w.writerows(per_doc_rows)
    return metrics
