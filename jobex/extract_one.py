"""Швидка перевірка кроку 1: один текст -> Claude -> JSON -> валідація.

    python -m jobex.extract_one data/raw/job_15.txt [--model claude-haiku-5-5] [--prompt v1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clients import Sampling, make_client
from pydantic import ValidationError

from .schema import JobPosting, json_schema
from .prompts import load_prompt


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
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--prompt", default="v1")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    client = make_client({"id": args.model, "provider": "anthropic", "model": args.model,
                          "supports_sampling": False}, {"timeout_s": 60, "max_retries": 3})
    prompt = load_prompt(args.prompt)
    text = Path(args.path).read_text(encoding="utf-8")
    resp = client.extract(prompt.system, prompt.render_user(text), json_schema(), Sampling("api_default"))
    obj, errors = validate(resp.data)
    print(json.dumps(resp.data, ensure_ascii=False, indent=2))
    print(f"\nvalid={obj is not None} attempts={resp.attempts} latency={resp.latency_ms:.0f}ms "
          f"tokens={resp.input_tokens}/{resp.output_tokens} mode={resp.sampling_sent.get('structured_mode')}")
    if errors:
        print("validation errors:\n" + errors)


if __name__ == "__main__":
    main()
