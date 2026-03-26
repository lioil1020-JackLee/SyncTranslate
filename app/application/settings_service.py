from __future__ import annotations

from pathlib import Path
import tempfile

from app.bootstrap.runtime_paths import runtime_snapshots_dir
from app.infra.config.schema import AppConfig
from app.infra.config.settings_store import load_config, save_config


class SettingsService:
    def __init__(self, config_path: str) -> None:
        self._config_path = config_path

    @property
    def config_path(self) -> str:
        return self._config_path

    def load(self) -> AppConfig:
        return load_config(self._config_path)

    def save(self, config: AppConfig) -> Path:
        return save_config(config, self._config_path)

    def save_snapshot(
        self,
        config: AppConfig,
        *,
        prefix: str = "healthcheck_",
        suffix: str = ".yaml",
        snapshot_dir: str | None = None,
    ) -> str:
        target_dir = Path(snapshot_dir) if snapshot_dir else runtime_snapshots_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            prefix=prefix,
            dir=str(target_dir),
            delete=False,
            encoding="utf-8",
        ) as fp:
            save_config(config, fp.name)
            return fp.name


__all__ = ["SettingsService"]
