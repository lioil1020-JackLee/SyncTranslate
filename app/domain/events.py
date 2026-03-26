from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class PipelineMeta:
    source: str
    utterance_id: str
    revision: int
    created_at: float


@dataclass(slots=True)
class ErrorEvent:
    level: str
    module: str
    code: str
    message: str
    source: str | None = None
    detail: str | None = None
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at <= 0:
            self.created_at = time.time()

    def to_log_line(self) -> str:
        scope = self.module if not self.source else f"{self.module}:{self.source}"
        detail = f" | {self.detail}" if self.detail else ""
        return f"[{self.level}] {scope} {self.code} - {self.message}{detail}"


__all__ = ["PipelineMeta", "ErrorEvent"]
