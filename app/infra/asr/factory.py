from __future__ import annotations

from typing import Callable

from app.domain.events import ErrorEvent
from app.infra.asr.contracts import AsrManagerProtocol
from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.config.schema import AppConfig


def normalize_asr_pipeline_mode(value: str) -> str:
    normalized = (value or "v2").strip().lower()
    if normalized in {"legacy", "v1", "whisper_legacy", "v2", "next", "asr_v2"}:
        return "v2"
    return "v2"


def create_asr_manager(
    config: AppConfig,
    on_error: Callable[[str | ErrorEvent], None] | None = None,
    *,
    pipeline_revision: int = 1,
) -> AsrManagerProtocol:
    return ASRManagerV2(config, on_error=on_error, pipeline_revision=pipeline_revision)


__all__ = ["create_asr_manager", "normalize_asr_pipeline_mode"]
