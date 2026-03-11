from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.schemas import AppConfig


def load_config(config_path: str | Path = "config.yaml", fallback_path: str | Path = "config.example.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        path = Path(fallback_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        raw: dict[str, Any] = yaml.safe_load(fp) or {}
    return AppConfig.from_dict(raw)


def save_config(config: AppConfig, config_path: str | Path = "config.yaml") -> Path:
    path = Path(config_path)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config.to_dict(), fp, sort_keys=False, allow_unicode=True)
    return path
