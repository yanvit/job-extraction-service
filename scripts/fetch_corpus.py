#!/usr/bin/env python3
"""Завантажує довідкові документи з data/sources/manifest.yaml у data/docs/.

    python scripts/fetch_corpus.py            # завантажити відсутні
    python scripts/fetch_corpus.py --force    # перезавантажити все

Ввічливо: пауза між запитами, User-Agent із контактом проєкту, жодних паралельних запитів.
Результат: data/docs/<id>.<html|pdf> + data/docs/manifest.lock.json (sha256, розмір, дата, HTTP-статус).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "sources" / "manifest.yaml"
OUT = ROOT / "data" / "docs"
LOCK = OUT / "manifest.lock.json"
HEADERS = {"User-Agent": "job-extraction-service/0.2 (educational RAG project; github.com/yanvit/job-extraction-service)"}
PAUSE_S = 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    docs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    failed = []

    for i, d in enumerate(docs, 1):
        path = OUT / f"{d['id']}.{d['format']}"
        if path.exists() and not args.force:
            print(f"[{i}/{len(docs)}] {d['id']}: already here, skip")
            continue
        try:
            r = requests.get(d["url"], headers=HEADERS, timeout=60)
            r.raise_for_status()
            body = r.content
            if d["format"] == "pdf" and not body.startswith(b"%PDF"):
                raise ValueError("expected a PDF, got something else (login page / blocked?)")
            if d["format"] == "html" and body.lstrip().startswith(b"%PDF"):
                raise ValueError("URL returns a PDF — set format: pdf in the manifest")
            path.write_bytes(body)
            lock[d["id"]] = {
                "url": d["url"], "file": path.name, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(), "http_status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"[{i}/{len(docs)}] {d['id']}: {len(body) // 1024} KB")
        except Exception as e:  # noqa: BLE001 — один недоступний документ не зупиняє решту
            failed.append(d["id"])
            print(f"[{i}/{len(docs)}] {d['id']}: FAILED — {e}")
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
        if i < len(docs):
            time.sleep(PAUSE_S)

    print(f"\n{len(docs) - len(failed)}/{len(docs)} documents available in {OUT.relative_to(ROOT)}")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
