from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable

import numpy as np

from app.model_providers import AsrProvider


@dataclass(slots=True)
class AsrResult:
    text: str
    is_final: bool


class AsrWorker:
    def __init__(self, asr_provider: AsrProvider) -> None:
        self._asr_provider = asr_provider
        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=64)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._on_result: Callable[[AsrResult], None] | None = None
        self._speech_seconds = 0.0
        self._segment_count = 0
        self._partial_emitted = False
        self._segment_audio_chunks: list[np.ndarray] = []

    def start(self, on_result: Callable[[AsrResult], None]) -> None:
        self.stop()
        self._on_result = on_result
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._speech_seconds = 0.0
        self._segment_count = 0
        self._partial_emitted = False
        self._segment_audio_chunks = []
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def submit_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        try:
            self._queue.put_nowait((chunk, sample_rate))
        except Exception:
            return

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk, sample_rate = self._queue.get(timeout=0.2)
            except Empty:
                continue

            if sample_rate <= 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            duration_sec = float(len(chunk) / sample_rate)
            if rms > 0.01:
                self._speech_seconds += duration_sec
                self._segment_audio_chunks.append(chunk)

            if self._speech_seconds >= 0.4 and not self._partial_emitted:
                if self._on_result:
                    self._on_result(AsrResult(text=self._asr_provider.partial_text(), is_final=False))
                self._partial_emitted = True

            if self._speech_seconds >= 1.2:
                self._segment_count += 1
                segment_audio = (
                    np.concatenate(self._segment_audio_chunks, axis=0)
                    if self._segment_audio_chunks
                    else np.zeros((0, 1), dtype=np.float32)
                )
                try:
                    text = self._asr_provider.final_text(
                        audio=segment_audio,
                        sample_rate=int(sample_rate),
                        segment_index=self._segment_count,
                    )
                    if self._on_result:
                        self._on_result(AsrResult(text=text, is_final=True))
                except Exception as exc:
                    if self._on_result:
                        self._on_result(AsrResult(text=f"[asr-error] {exc}", is_final=True))
                self._speech_seconds = 0.0
                self._partial_emitted = False
                self._segment_audio_chunks = []
