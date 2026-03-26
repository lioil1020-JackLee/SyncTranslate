from __future__ import annotations

from pathlib import Path

from app.application.diagnostics_export import export_runtime_diagnostics, export_session_report
from app.infra.config.schema import AppConfig


class ExportService:
    def export_runtime(
        self,
        *,
        config_path: str,
        config: AppConfig,
        routes,
        runtime_stats_text: str,
        recent_errors: list[str],
    ) -> Path:
        return export_runtime_diagnostics(
            config_path=config_path,
            config=config,
            routes=routes,
            runtime_stats_text=runtime_stats_text,
            recent_errors=recent_errors,
        )

    def export_session(
        self,
        *,
        config_path: str,
        config: AppConfig,
        routes,
        payload: dict[str, object],
        recent_errors: list[str],
    ) -> Path:
        return export_session_report(
            config_path=config_path,
            config=config,
            routes=routes,
            payload=payload,
            recent_errors=recent_errors,
        )


__all__ = ["ExportService"]
