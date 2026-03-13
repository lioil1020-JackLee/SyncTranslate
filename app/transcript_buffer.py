from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock


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


class TranscriptBuffer:
    def __init__(self, max_items: int = 200) -> None:
        self._items: deque[TranscriptItem] = deque(maxlen=max_items)
        self._lock = Lock()

    def upsert_event(
        self,
        source: str,
        channel: str,
        kind: str,
        text: str,
        is_final: bool,
        *,
        utterance_id: str | None = None,
        revision: int = 0,
        latency_ms: int | None = None,
        created_at: datetime | None = None,
    ) -> None:
        item = TranscriptItem(
            source=source,
            channel=channel,
            kind=kind,
            utterance_id=utterance_id,
            revision=max(0, int(revision)),
            text=text,
            is_final=is_final,
            latency_ms=None if latency_ms is None else int(latency_ms),
            created_at=created_at or datetime.now(),
        )
        with self._lock:
            if item.utterance_id:
                if self._upsert_by_utterance_locked(item):
                    return
            self._remove_latest_partial_locked(source)
            self._items.append(item)

    def append(
        self,
        source: str,
        text: str,
        is_final: bool,
        *,
        utterance_id: str | None = None,
        revision: int = 0,
        created_at: datetime | None = None,
    ) -> None:
        # Backward-compatible helper for existing call sites.
        self.upsert_event(
            source=source,
            channel=source,
            kind="caption",
            text=text,
            is_final=is_final,
            utterance_id=utterance_id,
            revision=revision,
            created_at=created_at,
        )

    def _upsert_by_utterance_locked(self, item: TranscriptItem) -> bool:
        items = list(self._items)
        for idx in range(len(items) - 1, -1, -1):
            current = items[idx]
            if current.source != item.source or current.utterance_id != item.utterance_id:
                continue
            if item.revision and current.revision and item.revision < current.revision:
                return True
            if current.is_final and not item.is_final:
                return True
            items[idx] = item
            self._items = deque(items, maxlen=self._items.maxlen)
            return True
        return False

    def _remove_latest_partial_locked(self, source: str) -> None:
        items = list(self._items)
        for idx in range(len(items) - 1, -1, -1):
            current = items[idx]
            if current.source == source and not current.is_final:
                items.pop(idx)
                self._items = deque(items, maxlen=self._items.maxlen)
                return

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def latest(self, source: str, limit: int = 20) -> list[TranscriptItem]:
        with self._lock:
            items = [item for item in self._items if item.source == source]
        return items[-limit:]
