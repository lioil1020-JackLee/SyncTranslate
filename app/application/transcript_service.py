from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
import re

from app.domain.transcript_models import TranscriptItem


_SENTENCE_END_RE = re.compile(r'[.?!。！？…]["\')\]\u300d\u300f\u3011\u300b\u3009]*$')
_OPENING_PUNCTUATION = "\"'([{<\u300c\u300e\u3010\u300a\u3008"
_NEW_SENTENCE_PREFIXES = (
    "However",
    "But",
    "And",
    "So",
    "Then",
    "Meanwhile",
    "因為",
    "所以",
    "但是",
    "然後",
    "另外",
    "接著",
)


class TranscriptService:
    def __init__(self, max_items: int | None = 200) -> None:
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
        is_stable_partial: bool = False,
        latency_ms: int | None = None,
        created_at: datetime | None = None,
        speaker_label: str = "",
    ) -> None:
        item = TranscriptItem(
            source=source,
            channel=channel,
            kind=kind,
            utterance_id=utterance_id,
            revision=max(0, int(revision)),
            text=text,
            is_final=is_final,
            is_stable_partial=bool(is_stable_partial and not is_final),
            latency_ms=None if latency_ms is None else int(latency_ms),
            created_at=created_at or datetime.now(),
            speaker_label=str(speaker_label or ""),
        )
        with self._lock:
            if item.utterance_id and self._upsert_by_utterance_locked(item):
                return
            self._remove_latest_partial_locked(source)
            if self._merge_with_previous_final_locked(item):
                return
            self._items.append(item)

    def append(
        self,
        source: str,
        text: str,
        is_final: bool,
        *,
        utterance_id: str | None = None,
        revision: int = 0,
        is_stable_partial: bool = False,
        created_at: datetime | None = None,
        speaker_label: str = "",
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
            is_stable_partial=is_stable_partial,
            created_at=created_at,
            speaker_label=speaker_label,
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

    def _merge_with_previous_final_locked(self, item: TranscriptItem) -> bool:
        if not item.is_final:
            return False
        items = list(self._items)
        if not items:
            return False
        previous = items[-1]
        if not self._should_merge_adjacent_finals(previous, item):
            return False
        merged_text = self._join_transcript_text(previous.text, item.text)
        items[-1] = TranscriptItem(
            source=previous.source,
            channel=previous.channel,
            kind=previous.kind,
            utterance_id=previous.utterance_id,
            revision=max(previous.revision, item.revision),
            text=merged_text,
            is_final=True,
            is_stable_partial=False,
            latency_ms=item.latency_ms if item.latency_ms is not None else previous.latency_ms,
            created_at=item.created_at,
            speaker_label=previous.speaker_label,
        )
        self._items = deque(items, maxlen=self._items.maxlen)
        return True

    @staticmethod
    def _should_merge_adjacent_finals(previous: TranscriptItem, current: TranscriptItem) -> bool:
        if not previous.is_final or not current.is_final:
            return False
        if previous.source != current.source or previous.channel != current.channel or previous.kind != current.kind:
            return False
        if previous.speaker_label != current.speaker_label:
            return False
        if previous.utterance_id and current.utterance_id and previous.utterance_id != current.utterance_id:
            return False
        previous_text = previous.text.strip()
        current_text = current.text.strip()
        if not previous_text or not current_text:
            return False
        if TranscriptService._ends_sentence(previous_text):
            return False
        if TranscriptService._starts_new_sentence(current_text):
            return False
        gap_ms = max(0, int((current.created_at - previous.created_at).total_seconds() * 1000))
        if gap_ms > 1600:
            return False
        combined_len = len(previous_text) + len(current_text)
        if combined_len > 60 and len(current_text) > 12 and len(previous_text) > 20:
            return False
        if len(current_text) <= 12 or len(previous_text) <= 20:
            return gap_ms <= 400
        return gap_ms <= 700

    @staticmethod
    def _join_transcript_text(previous: str, current: str) -> str:
        left = previous.rstrip()
        right = current.lstrip()
        if not left:
            return right
        if not right:
            return left
        if TranscriptService._needs_space_between(left[-1], right[0]):
            return f"{left} {right}"
        return f"{left}{right}"

    @staticmethod
    def _needs_space_between(left: str, right: str) -> bool:
        return left.isascii() and left.isalnum() and right.isascii() and right.isalnum()

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        return bool(_SENTENCE_END_RE.search(text.strip()))

    @staticmethod
    def _starts_new_sentence(text: str) -> bool:
        value = text.strip().lstrip(_OPENING_PUNCTUATION)
        if not value:
            return False
        if value[0].isupper():
            return True
        return any(value.startswith(prefix) for prefix in _NEW_SENTENCE_PREFIXES)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def latest(self, source: str, limit: int = 20) -> list[TranscriptItem]:
        with self._lock:
            items = [item for item in self._items if item.source == source]
        return items[-limit:]


class TranscriptBuffer(TranscriptService):
    """Compatibility alias during migration from buffer -> service naming."""


__all__ = ["TranscriptItem", "TranscriptService", "TranscriptBuffer"]
