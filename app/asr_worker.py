from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.local_ai.streaming_asr import AsrEvent, StreamingAsr
from app.local_ai.vad_segmenter import VadConfig, VadSegmenter


@dataclass(slots=True)
class AsrResult:
    text: str
    is_final: bool


class _ProviderAdapter:
    def __init__(self, provider) -> None:
        self._provider = provider
        self._segment_index = 0

    def transcribe_partial(self, audio: np.ndarray, sample_rate: int) -> str:
        partial = getattr(self._provider, "partial_text", None)
        if callable(partial):
            return str(partial())
        return ""

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> str:
        final = getattr(self._provider, "final_text", None)
        if not callable(final):
            return ""
        self._segment_index += 1
        return str(final(audio=audio, sample_rate=sample_rate, segment_index=self._segment_index))


class AsrWorker:
    def __init__(self, asr_provider) -> None:
        self._stream = StreamingAsr(
            engine=_ProviderAdapter(asr_provider),
            vad=VadSegmenter(VadConfig()),
            partial_interval_ms=400,
        )
        self._on_result: Callable[[AsrResult], None] | None = None

    def start(self, on_result: Callable[[AsrResult], None]) -> None:
        self._on_result = on_result
        self._stream.start(self._handle_event)

    def stop(self) -> None:
        self._stream.stop()

    def submit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._stream.submit_chunk(chunk, sample_rate)

    def _handle_event(self, event: AsrEvent) -> None:
        if self._on_result:
            self._on_result(AsrResult(text=event.text, is_final=event.is_final))
