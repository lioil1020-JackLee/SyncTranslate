"""TTSDispatcher — manages TTS enqueue, events, and playback metrics.

Extracted from AudioRouter to make TTS dispatch logic independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.domain.constants import OUTPUT_MODE_TTS
from app.infra.tts.playback_queue import TTSManager


@dataclass
class TTSEnqueueRequest:
    """All data required to enqueue one TTS utterance."""
    channel: str
    text: str
    utterance_id: str
    revision: int
    is_final: bool
    is_stable_partial: bool
    is_early_final: bool


class TTSDispatcher:
    """Encapsulates TTS enqueue logic and on_tts_request event emission.

    Parameters
    ----------
    tts_manager:
        The underlying TTS playback queue manager.
    on_tts_request:
        Optional callback called before TTS is enqueued (e.g. for diagnostics).
    """

    def __init__(
        self,
        *,
        tts_manager: TTSManager,
        on_tts_request: Callable[[str, str], None] | None = None,
    ) -> None:
        self._tts = tts_manager
        self._on_tts_request = on_tts_request
        self._enqueue_count: int = 0
        self._skipped_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_enqueue(self, req: TTSEnqueueRequest) -> bool:
        """Enqueue TTS if the channel's output mode is TTS and text is non-empty.

        Returns True if TTS was enqueued, False otherwise.
        """
        if self._tts.output_mode(req.channel) != OUTPUT_MODE_TTS:
            self._skipped_count += 1
            return False
        text = req.text.strip()
        if not text:
            self._skipped_count += 1
            return False
        if self._on_tts_request:
            self._on_tts_request(req.channel, text)
        self._tts.enqueue(
            req.channel,
            text,
            utterance_id=req.utterance_id,
            revision=req.revision,
            is_final=req.is_final,
            is_stable_partial=req.is_stable_partial,
            is_early_final=req.is_early_final,
        )
        self._enqueue_count += 1
        return True

    def stats(self) -> dict[str, int]:
        return {
            "enqueue_count": self._enqueue_count,
            "skipped_count": self._skipped_count,
        }


__all__ = ["TTSDispatcher", "TTSEnqueueRequest"]
