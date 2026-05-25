from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from app.version import app_version


@dataclass(frozen=True, slots=True)
class BuildMetadata:
    app_version: str
    config_schema_version: int
    git_commit: str
    build_timestamp: str
    runtime_mode: str
    packaged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "app_version": self.app_version,
            "config_schema_version": self.config_schema_version,
            "git_commit": self.git_commit,
            "build_timestamp": self.build_timestamp,
            "runtime_mode": self.runtime_mode,
            "packaged": self.packaged,
        }


def build_metadata(*, config_schema_version: int = 0, runtime_mode: str = "") -> BuildMetadata:
    return BuildMetadata(
        app_version=app_version(),
        config_schema_version=int(config_schema_version or 0),
        git_commit=_git_commit(),
        build_timestamp=os.environ.get("SYNCTRANSLATE_BUILD_TIMESTAMP", "").strip() or datetime.now(timezone.utc).isoformat(),
        runtime_mode=str(runtime_mode or ""),
        packaged=bool(getattr(sys, "frozen", False)),
    )


def _git_commit() -> str:
    env_commit = os.environ.get("SYNCTRANSLATE_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else "unknown"
    except Exception:
        return "unknown"


__all__ = ["BuildMetadata", "build_metadata"]

