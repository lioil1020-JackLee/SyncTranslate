from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.infra.config.schema import AppConfig


@dataclass(slots=True)
class ASREventWithSource:
    source: str
    utterance_id: str
    revision: int
    pipeline_revision: int
    config_fingerprint: str
    created_at: float
    text: str
    is_final: bool
    is_early_final: bool
    start_ms: int
    end_ms: int
    latency_ms: int
    detected_language: str
    raw_text: str = ""
    correction_applied: bool = False
    speaker_label: str = ""


class AsrManagerProtocol(Protocol):
    pipeline_mode: str

    def configure_pipeline(self, config: AppConfig, pipeline_revision: int) -> None: ...

    def refresh_runtime(self) -> None: ...

    def start(self, source: str, on_event) -> None: ...

    def stop(self, source: str) -> None: ...

    def stop_all(self) -> None: ...

    def submit(self, source: str, chunk: np.ndarray, sample_rate: float) -> None: ...

    def set_enabled(self, source: str, enabled: bool) -> None: ...

    def is_enabled(self, source: str) -> bool: ...

    def stats(self) -> dict[str, dict[str, object]]: ...


__all__ = ["ASREventWithSource", "AsrManagerProtocol"]
