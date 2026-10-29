"""Швидка перевірка однієї моделі на одному тексті: текст -> модель -> JSON -> валідація.

    python -m jobex.extract_one data/raw/job_15.txt                          # Claude Haiku 5.5, промпт v1
    python -m jobex.extract_one data/raw/job_15.txt --model qwen2.5-3b-local --sampling t0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from .clients import Sampling, make_client
from .prompts import load_prompt
from .schema import JobPosting, inline_refs, json_schema

ROOT = Path(__file__).resolve().parent.parent


def validate(data):
    if data is None:
        return None, "response is not a JSON object"
    try:
        return JobPosting.model_validate(data), None
    except ValidationError as e:
        return None, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="claude-haiku-5-5", help="id моделі з config.yaml")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--sampling", default=None, help="назва профілю з config.yaml (t0, t07_p09)")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mcfg = next((m for m in cfg["models"] if m["id"] == args.model), None)
    if mcfg is None:
        raise SystemExit(f"model '{args.model}' not in config.yaml: {[m['id'] for m in cfg['models']]}")
    client = make_client(mcfg, cfg["defaults"])

    if args.sampling and mcfg.get("supports_sampling", True):
        s = next(s for s in cfg["sampling"] if s["name"] == args.sampling)
        sampling = Sampling(s["name"], s.get("temperature"), s.get("top_p"))
    else:
        sampling = Sampling("api_default")

    schema = inline_refs(json_schema()) if mcfg["provider"] == "ollama" else json_schema()
    prompt = load_prompt(args.prompt)
    text = Path(args.path).read_text(encoding="utf-8")
    resp = client.extract(prompt.system, prompt.render_user(text), schema, sampling)
    obj, errors = validate(resp.data)
    print(json.dumps(resp.data, ensure_ascii=False, indent=2) if resp.data else resp.raw_text)
    print(f"\nmodel={mcfg['model']} valid={obj is not None} attempts={resp.attempts} "
          f"latency={resp.latency_ms:.0f}ms tokens={resp.input_tokens}/{resp.output_tokens} "
          f"sampling_sent={resp.sampling_sent}")
    if errors:
        print("validation errors:\n" + errors)


if __name__ == "__main__":
    main()
