from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TranscriptItem:
    source: str
    text: str
    is_final: bool
    created_at: datetime


class TranscriptBuffer:
    def __init__(self, max_items: int = 200) -> None:
        self._items: deque[TranscriptItem] = deque(maxlen=max_items)

    def append(self, source: str, text: str, is_final: bool) -> None:
        item = TranscriptItem(source=source, text=text, is_final=is_final, created_at=datetime.now())
        if self._items and self._items[-1].source == source and not self._items[-1].is_final:
            self._items.pop()
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def latest(self, source: str, limit: int = 20) -> list[TranscriptItem]:
        items = [item for item in self._items if item.source == source]
        return items[-limit:]
