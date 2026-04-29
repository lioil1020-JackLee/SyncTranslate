"""Pipeline latency tracker extracted from AudioRouter.

Records ASR / translation / TTS-enqueue / playback-start timestamps keyed by
(source, utterance_id) and maintains a rolling window of completed entries.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime


class PipelineLatencyTracker:
    """Accumulate per-utterance latency milestones.

    Parameters
    ----------
    tts_manager:
        The TTSManager instance; used in :meth:`record_playback_start` to look
        up the currently-playing TTS task.
    maxlen:
        Maximum number of completed utterance records to keep.
    """

    def __init__(self, tts_manager=None, *, maxlen: int = 32) -> None:
        self._tts_manager = tts_manager
        self._by_utterance: dict[tuple[str, str], dict[str, object]] = {}
        self._recent: deque[dict[str, object]] = deque(maxlen=maxlen)

    def set_tts_manager(self, tts_manager) -> None:
        self._tts_manager = tts_manager

    def reset(self) -> None:
        self._by_utterance.clear()
        self._recent.clear()

    def recent(self) -> list[dict[str, object]]:
        return list(self._recent)

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_asr(self, event) -> None:
        """Record ASR partial/final timing for *event*."""
        key = (event.source, event.utterance_id)
        entry = self._by_utterance.setdefault(
            key,
            {
                "source": event.source,
                "utterance_id": event.utterance_id,
                "revision": event.revision,
                "speech_start_ms": event.start_ms,
            },
        )
        entry["revision"] = event.revision
        entry["speech_start_ms"] = min(
            int(entry.get("speech_start_ms", event.start_ms)), int(event.start_ms)
        )
        if not event.is_final and "first_asr_partial_ms" not in entry:
            entry["first_asr_partial_ms"] = max(0, int(event.end_ms) - int(event.start_ms))
        if event.is_final:
            entry["speech_end_to_asr_final_ms"] = max(0, int(event.latency_ms))
            entry["asr_final_kind"] = "early_final" if event.is_early_final else "final"

    def record_translation(self, event, translated: object) -> None:
        """Record the moment a translation result became ready."""
        key = (event.source, event.utterance_id)
        entry = self._by_utterance.setdefault(
            key, {"source": event.source, "utterance_id": event.utterance_id}
        )
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["translation_ready_at_ms"] = now_ms
        if not event.is_final and "first_display_partial_ms" not in entry:
            entry["first_display_partial_ms"] = max(0, int(event.end_ms) - int(event.start_ms))
        if getattr(translated, "is_final", False):
            entry["asr_final_to_llm_final_ms"] = max(0, now_ms - int(event.created_at * 1000))

    def record_tts_enqueue(
        self,
        *,
        channel: str,
        source: str,
        utterance_id: str,
        revision: int,
        is_final: bool,
        is_stable_partial: bool,
        is_early_final: bool,
    ) -> None:
        """Record the moment a TTS synthesis request was enqueued."""
        key = (source, utterance_id)
        entry = self._by_utterance.setdefault(
            key, {"source": source, "utterance_id": utterance_id}
        )
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["tts_channel"] = channel
        entry["tts_enqueue_at_ms"] = now_ms
        entry["tts_enqueue_revision"] = revision
        entry["tts_enqueue_kind"] = (
            "final"
            if is_final
            else "early_final"
            if is_early_final
            else "stable_partial"
            if is_stable_partial
            else "partial"
        )

    def record_playback_start(self, channel: str) -> None:
        """Record playback start and push a completed entry to the rolling window."""
        current = getattr(self._tts_manager, "current_task", lambda _ch: None)(channel)
        if not current:
            return
        source = "remote" if channel == "local" else "local"
        utterance_id = str(current.get("utterance_id") or "")
        if not utterance_id:
            return
        key = (source, utterance_id)
        entry = self._by_utterance.setdefault(
            key, {"source": source, "utterance_id": utterance_id}
        )
        now_ms = int(datetime.now().timestamp() * 1000)
        entry["playback_start_at_ms"] = now_ms
        enqueue_at = int(entry.get("tts_enqueue_at_ms", now_ms))
        entry["tts_enqueue_to_playback_start_ms"] = max(0, now_ms - enqueue_at)
        self._recent.appendleft(
            {
                "source": entry.get("source", source),
                "utterance_id": entry.get("utterance_id", utterance_id),
                "first_asr_partial_ms": entry.get("first_asr_partial_ms"),
                "first_display_partial_ms": entry.get("first_display_partial_ms"),
                "speech_end_to_asr_final_ms": entry.get("speech_end_to_asr_final_ms"),
                "asr_final_to_llm_final_ms": entry.get("asr_final_to_llm_final_ms"),
                "tts_enqueue_to_playback_start_ms": entry.get("tts_enqueue_to_playback_start_ms"),
                "tts_enqueue_kind": entry.get("tts_enqueue_kind", "unknown"),
            }
        )


__all__ = ["PipelineLatencyTracker"]
