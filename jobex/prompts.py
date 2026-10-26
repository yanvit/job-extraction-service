"""Завантаження версіонованих промптів з prompts/<version>/."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str
    repair_template: str
    meta: dict

    def render_user(self, text: str) -> str:
        return self.user_template.replace("{{TEXT}}", text)

    def render_repair(self, text: str, previous: str, errors: str) -> str:
        return (self.repair_template.replace("{{TEXT}}", text)
                .replace("{{PREVIOUS}}", previous).replace("{{ERRORS}}", errors))

    @property
    def sha(self) -> str:
        h = hashlib.sha256((self.system + self.user_template + self.repair_template).encode()).hexdigest()
        return h[:12]


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    d = prompts_dir / version
    if not d.is_dir():
        raise FileNotFoundError(f"prompt version '{version}' not found in {prompts_dir}")
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8")) or {}
    if meta.get("version") != version:
        raise ValueError(f"{d}/meta.yaml: version must be '{version}'")
    return Prompt(version=version,
                  system=(d / "system.md").read_text(encoding="utf-8").strip(),
                  user_template=(d / "user.md").read_text(encoding="utf-8").strip(),
                  repair_template=(d / "repair.md").read_text(encoding="utf-8").strip(),
                  meta=meta)
