from __future__ import annotations

from pathlib import Path


def resolve_runtime_path(path_text: str) -> Path:
    raw = (path_text or "").strip()
    if not raw:
        return Path("")
    return Path(raw).expanduser().resolve()


def path_exists(path_text: str) -> bool:
    path = resolve_runtime_path(path_text)
    return bool(path) and path.exists()
