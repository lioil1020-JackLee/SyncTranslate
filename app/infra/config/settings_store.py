from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml

from app.infra.config.schema import AppConfig
from app.infra.config._config_migration import (  # noqa: F401
    migrate_legacy_config,
    is_legacy_config,
    _normalize_asr_engine_name,
    _normalize_vad_backend_name,
    _normalize_asr_profile_legacy_fields,
)
from app.infra.config._config_serialization import (  # noqa: F401
    _normalize_external_config_keys,
    _present_external_config_keys,
)



def _runtime_base_dirs() -> list[Path]:
    dirs: list[Path] = [Path.cwd()]

    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(item)
    return unique_dirs


def _resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in _runtime_base_dirs():
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def _resolve_write_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    existing = _resolve_existing_path(path)
    if existing.exists():
        return existing
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / path
    return Path.cwd() / path


def _ensure_config_file(config_path: str | Path = "config.yaml") -> Path:
    target = _resolve_write_path(config_path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _present_external_config_keys(AppConfig().to_dict())
    with target.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return target


def load_config(config_path: str | Path = "config.yaml", fallback_path: str | Path = "config.example.yaml") -> AppConfig:
    path = _resolve_existing_path(config_path)
    if not path.exists():
        fallback = _resolve_existing_path(fallback_path)
        if fallback.exists():
            path = fallback
        else:
            path = _ensure_config_file(config_path)

    with path.open("r", encoding="utf-8") as fp:
        raw: dict[str, Any] = yaml.safe_load(fp) or {}
    raw = _normalize_external_config_keys(raw)

    migrated = migrate_legacy_config(raw)
    config = AppConfig.from_dict(migrated)
    return config


def save_config(config: AppConfig, config_path: str | Path = "config.yaml") -> Path:
    path = _resolve_write_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _present_external_config_keys(config.to_dict())
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return path

