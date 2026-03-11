from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock


@dataclass(slots=True)
class TranscriptItem:
    source: str
    text: str
    is_final: bool
    created_at: datetime


class TranscriptBuffer:
    def __init__(self, max_items: int = 200) -> None:
        self._items: deque[TranscriptItem] = deque(maxlen=max_items)
        self._lock = Lock()

    def append(self, source: str, text: str, is_final: bool) -> None:
        item = TranscriptItem(source=source, text=text, is_final=is_final, created_at=datetime.now())
        with self._lock:
            if self._items and self._items[-1].source == source and not self._items[-1].is_final:
                self._items.pop()
            self._items.append(item)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def latest(self, source: str, limit: int = 20) -> list[TranscriptItem]:
        with self._lock:
            items = [item for item in self._items if item.source == source]
        return items[-limit:]
