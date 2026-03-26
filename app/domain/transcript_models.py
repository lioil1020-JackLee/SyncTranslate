from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TranscriptItem:
    source: str
    channel: str
    kind: str
    utterance_id: str | None
    revision: int
    text: str
    is_final: bool
    latency_ms: int | None
    created_at: datetime


__all__ = ["TranscriptItem"]
