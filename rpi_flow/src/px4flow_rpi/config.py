from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.json must be a JSON object")
        return AppConfig(raw=data)

    def section(self, name: str) -> dict[str, Any]:
        sec = self.raw.get(name, {})
        if not isinstance(sec, dict):
            raise ValueError(f"config section '{name}' must be an object")
        return sec

