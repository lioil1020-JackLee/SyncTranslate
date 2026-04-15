"""ASREventProcessor — handles ASR events and updates the transcript buffer.

This module extracts the transcript-update responsibility from AudioRouter,
making it independently testable and more focused.

AudioRouter delegates to this class after receiving a raw ASREventWithSource,
so the router itself stays thin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from app.application.transcript_postprocessor import TranscriptPostProcessor
from app.application.transcript_service import TranscriptBuffer
from app.infra.asr.contracts import ASREventWithSource
from app.infra.translation.engine import TranslatorManager


@dataclass
class ASREventProcessorConfig:
    """Configuration for ASREventProcessor behaviour."""
    max_latency_ms: float = 8000.0
    display_partial: str = "all"  # "all" | "stable_only" | "none"


class ASREventProcessor:
    """Process incoming ASR events: postprocess text → store in transcript buffer.

    This class intentionally has no I/O side-effects beyond writing to the
    transcript buffer and calling the optional callbacks.  All downstream
    actions (translation, TTS) are left to the caller.

    Parameters
    ----------
    transcript_buffer:
        Shared transcript buffer (one per session).
    postprocessor:
        Text post-processor (normalization, stabilization, glossary).
    translator_manager:
        Used only to look up channel mappings and check if translation is enabled.
    on_store:
        Optional callback called after each successful transcript store.
    config:
        Behavioural configuration.
    """

    def __init__(
        self,
        *,
        transcript_buffer: TranscriptBuffer,
        postprocessor: TranscriptPostProcessor,
        translator_manager: TranslatorManager,
        on_store: Callable[[str, str, bool], None] | None = None,
        config: ASREventProcessorConfig | None = None,
    ) -> None:
        self._buffer = transcript_buffer
        self._postprocessor = postprocessor
        self._translator = translator_manager
        self._on_store = on_store
        self._cfg = config or ASREventProcessorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, event: ASREventWithSource) -> bool:
        """Handle one ASR event.  Returns True if the event was processed.

        Steps
        -----
        1. Latency gate (drop if too old)
        2. Text postprocessing
        3. Store original transcript
        4. Return True so the caller can chain translation / TTS

        Parameters
        ----------
        event:
            Raw ASR event from ASRManager.

        Returns
        -------
        bool
            True if the event was stored; False if it was dropped.
        """
        if self._should_drop(event):
            return False

        detected_language = str(getattr(event, "detected_language", "") or "")
        if event.is_final:
            processed_text = self._postprocessor.process_final(
                event.source, event.text,
                language=detected_language,
                utterance_id=event.utterance_id or "",
            )
        else:
            processed_text = self._postprocessor.process_partial(
                event.source, event.text,
                language=detected_language,
                utterance_id=event.utterance_id or "",
            )

        original_channel = self._original_channel_of(event.source)
        self._store(
            channel=original_channel,
            kind="original",
            text=processed_text,
            event=event,
        )
        if self._on_store:
            self._on_store(original_channel, processed_text, event.is_final)
        return True

    def update_postprocessor(self, postprocessor: TranscriptPostProcessor) -> None:
        self._postprocessor = postprocessor

    def update_config(self, config: ASREventProcessorConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_drop(self, event: ASREventWithSource) -> bool:
        if self._cfg.max_latency_ms <= 0:
            return False
        latency_ms = float(getattr(event, "latency_ms", 0) or 0)
        return latency_ms > self._cfg.max_latency_ms

    def _original_channel_of(self, source: str) -> str:
        # Maintain the same channel mapping convention used by AudioRouter
        if source == "local":
            return "local"
        return "remote"

    def _store(self, *, channel: str, kind: str, text: str, event: ASREventWithSource) -> None:
        try:
            self._buffer.upsert(
                source=channel,
                channel=channel,
                kind=kind,
                text=text,
                is_final=event.is_final,
                is_stable_partial=not event.is_final,
                utterance_id=event.utterance_id,
                revision=event.revision,
                latency_ms=event.latency_ms,
                created_at=datetime.fromtimestamp(event.created_at),
                speaker_label=getattr(event, "speaker_label", ""),
            )
        except Exception:  # noqa: BLE001
            pass  # Transcript buffer errors must not crash the pipeline


__all__ = ["ASREventProcessor", "ASREventProcessorConfig"]
