from __future__ import annotations

from pathlib import Path
import tempfile


def runtime_root_dir() -> Path:
    return Path(tempfile.gettempdir()) / "SyncTranslate"


def runtime_logs_dir() -> Path:
    path = runtime_root_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_snapshots_dir() -> Path:
    path = runtime_root_dir() / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["runtime_logs_dir", "runtime_root_dir", "runtime_snapshots_dir"]
