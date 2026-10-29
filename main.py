#!/usr/bin/env python3
"""Structured Extraction Service — вакансії українського ринку -> JSON за схемою.

    python main.py            # = all: run -> eval -> report
    python main.py check      # перевірити ключі / Ollama
    python main.py run --models qwen2.5-3b-local --prompts v2 --docs golden
    python main.py eval
    python main.py report
    python main.py regress --model claude-haiku-5-5 --baseline v1 --candidate v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from jobex.clients import FatalError, Sampling, make_client  # noqa: E402
from jobex.pipeline import load_docs, load_golden_ids, run_experiment  # noqa: E402
from jobex.prompts import load_prompt  # noqa: E402


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv необов'язковий
        pass


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def model_cfg_by_id(cfg: dict, model_id: str) -> dict:
    for m in cfg["models"]:
        if m["id"] == model_id:
            return m
    raise SystemExit(f"model '{model_id}' is not defined in config.yaml")


def samplings_for(cfg: dict, model_cfg: dict, names: list[str]) -> list[Sampling]:
    by_name = {s["name"]: s for s in cfg["sampling"]}
    if model_cfg.get("supports_sampling", True) is False:
        # модель не приймає temperature/top_p -> немає сенсу ганяти кілька профілів
        return [Sampling(name="api_default")]
    return [Sampling(name=n, temperature=by_name[n].get("temperature"), top_p=by_name[n].get("top_p"))
            for n in names]


def cmd_run(cfg: dict, args) -> int:
    exp = cfg["experiment"]
    paths = cfg["paths"]
    models = args.models or exp["models"]
    prompts = args.prompts or exp["prompts"]
    sampling_names = args.sampling or exp["sampling"]
    docs_mode = args.docs or exp.get("docs", "all")
    only = load_golden_ids(ROOT / paths["golden"]) if docs_mode == "golden" else None
    docs = load_docs(ROOT / paths["data_dir"], only)
    use_cache = exp.get("cache", True) and not args.no_cache
    print(f"docs: {len(docs)} ({docs_mode}); models: {models}; prompts: {prompts}; sampling: {sampling_names}")

    ran = 0
    for model_id in models:
        mcfg = model_cfg_by_id(cfg, model_id)
        try:
            client = make_client(mcfg, cfg["defaults"])
        except FatalError as e:
            print(f"! skip {model_id}: {e}")
            continue
        conc = 1 if mcfg["provider"] == "ollama" else cfg["defaults"].get("concurrency", 1)
        for pv in prompts:
            prompt = load_prompt(pv)
            for s in samplings_for(cfg, mcfg, sampling_names):
                run_experiment(client, mcfg, prompt, s, docs, ROOT / paths["results_dir"],
                               use_cache=use_cache, concurrency=conc)
                ran += 1
    if ran == 0:
        print("nothing ran — check `python main.py check`")
        return 1
    return 0


def cmd_eval(cfg: dict, args) -> int:
    from jobex.evaluate import evaluate_all
    metrics = evaluate_all(ROOT / cfg["paths"]["results_dir"], ROOT / cfg["paths"]["golden"])
    if not metrics:
        print("no runs found in results/runs — run `python main.py run` first")
        return 1
    for m in sorted(metrics, key=lambda m: -(m["field_acc_overall"] or 0)):
        print(f'{m["run"]:<45} acc={m["field_acc_overall"] or 0:.3f} valid={m["valid_pct"]:.0f}% '
              f'p50={(m["latency_p50_ms"] or 0) / 1000:.1f}s p95={(m["latency_p95_ms"] or 0) / 1000:.1f}s '
              f'$/1k={m["cost_per_1000_docs_usd"]:.3f}')
    return 0


def cmd_report(cfg: dict, args) -> int:
    from jobex.report import update_report, write_golden_diff
    rd = ROOT / cfg["paths"]["results_dir"]
    mpath = rd / "metrics.json"
    if not mpath.exists():
        print("results/metrics.json not found — run `python main.py eval` first")
        return 1
    update_report(ROOT / "REPORT.md", json.loads(mpath.read_text(encoding="utf-8")))
    diff = write_golden_diff(rd, ROOT / cfg["paths"]["golden"])
    print(f"REPORT.md updated; per-field mismatches -> {diff.relative_to(ROOT)}")
    return 0


def cmd_regress(cfg: dict, args) -> int:
    """Регресійний прогін промптів на golden set: candidate не має бути гіршим за baseline."""
    from jobex.evaluate import evaluate_run, load_golden
    paths = cfg["paths"]
    golden = load_golden(ROOT / paths["golden"])
    docs = load_docs(ROOT / paths["data_dir"], golden.keys())
    mcfg = model_cfg_by_id(cfg, args.model)
    client = make_client(mcfg, cfg["defaults"])
    s = samplings_for(cfg, mcfg, [args.sampling])[0]
    res = {}
    for pv in (args.baseline, args.candidate):
        recs = run_experiment(client, mcfg, load_prompt(pv), s, docs, ROOT / paths["results_dir"],
                              concurrency=1 if mcfg["provider"] == "ollama" else cfg["defaults"].get("concurrency", 1))
        res[pv], _ = evaluate_run(recs, golden)
    b, c = res[args.baseline], res[args.candidate]
    print(f'\n{"field":<22}{args.baseline:>10}{args.candidate:>10}')
    worst = 0.0
    for key in ["field_acc_overall"] + [k for k in b if k.startswith("acc_")]:
        bv, cv = b[key] or 0, c[key] or 0
        flag = "  <-- regression" if cv < bv - args.max_field_drop else ""
        worst = min(worst, cv - bv) if key != "field_acc_overall" else worst
        print(f"{key:<22}{bv:>10.3f}{cv:>10.3f}{flag}")
    ok = (c["field_acc_overall"] or 0) >= (b["field_acc_overall"] or 0) - args.max_drop and worst >= -args.max_field_drop
    print("\nREGRESSION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def cmd_check(cfg: dict, args) -> int:
    import requests
    ok = True
    key = os.getenv("ANTHROPIC_API_KEY")
    print(f"ANTHROPIC_API_KEY: {'set' if key else 'MISSING (copy .env.example -> .env)'}")
    ok &= bool(key)
    for m in cfg["models"]:
        if m["provider"] != "ollama":
            continue
        base = (m.get("base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        try:
            tags = requests.get(f"{base}/api/tags", timeout=5).json()
            names = {t["name"] for t in tags.get("models", [])}
            have = m["model"] in names or f'{m["model"]}:latest' in names
            print(f"Ollama {base}: up; model {m['model']}: {'pulled' if have else 'MISSING -> ollama pull ' + m['model']}")
            ok &= have
        except requests.RequestException:
            print(f"Ollama {base}: NOT reachable -> install Ollama and run `ollama serve`")
            ok = False
    return 0 if ok else 1


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd")
    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+")
        p.add_argument("--prompts", nargs="+")
        p.add_argument("--sampling", nargs="+")
        p.add_argument("--docs", choices=["all", "golden"])
        p.add_argument("--no-cache", action="store_true")
    sub.add_parser("eval")
    sub.add_parser("report")
    sub.add_parser("check")
    p = sub.add_parser("regress")
    p.add_argument("--model", required=True)
    p.add_argument("--baseline", default="v1")
    p.add_argument("--candidate", default="v2")
    p.add_argument("--sampling", default="t0")
    p.add_argument("--max-drop", type=float, default=0.0, help="допустиме падіння загальної точності")
    p.add_argument("--max-field-drop", type=float, default=0.10, help="допустиме падіння по одному полю")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.cmd in (None, "all"):
        if args.cmd is None:
            args = ap.parse_args(["all"])
        rc = cmd_run(cfg, args)
        if rc:
            return rc
        return cmd_eval(cfg, args) or cmd_report(cfg, args)
    return {"run": cmd_run, "eval": cmd_eval, "report": cmd_report,
            "regress": cmd_regress, "check": cmd_check}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
