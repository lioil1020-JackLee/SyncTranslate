from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable

import numpy as np

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.local_ai.piper_tts import PiperTtsEngine
from app.local_ai.streaming_asr import AsrEvent, StreamingAsr
from app.local_ai.translation_stitcher import TranslationStitcher
from app.transcript_buffer import TranscriptBuffer


class DirectionalPipeline:
    def __init__(
        self,
        *,
        audio_capture: AudioCapture,
        transcript_buffer: TranscriptBuffer,
        audio_playback: AudioPlayback,
        asr_stream: StreamingAsr,
        stitcher: TranslationStitcher,
        tts: PiperTtsEngine,
        source_channel: str,
        translated_channel: str,
        get_output_device: Callable[[], str],
        tts_sample_rate: int,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._audio_capture = audio_capture
        self._transcript_buffer = transcript_buffer
        self._audio_playback = audio_playback
        self._asr_stream = asr_stream
        self._stitcher = stitcher
        self._tts = tts
        self._source_channel = source_channel
        self._translated_channel = translated_channel
        self._get_output_device = get_output_device
        self._tts_sample_rate = tts_sample_rate
        self._on_error = on_error
        self._running = False
        self._tts_queue: Queue[str] = Queue(maxsize=64)
        self._tts_stop = Event()
        self._tts_thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self, input_device_name: str, sample_rate: int, chunk_ms: int = 100) -> None:
        self.stop()
        self._asr_stream.start(self._handle_asr_event)
        self._audio_capture.add_consumer(self._on_audio_chunk)
        self._start_tts_worker()
        self._audio_capture.start(input_device_name, sample_rate=sample_rate, chunk_ms=chunk_ms)
        self._running = True

    def stop(self) -> None:
        self._audio_capture.remove_consumer(self._on_audio_chunk)
        self._audio_capture.stop()
        self._asr_stream.stop()
        self._audio_playback.stop()
        self._stop_tts_worker()
        self._running = False

    def _on_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._asr_stream.submit_chunk(chunk=chunk, sample_rate=sample_rate)

    def _handle_asr_event(self, event: AsrEvent) -> None:
        try:
            self._transcript_buffer.append(source=self._source_channel, text=event.text, is_final=event.is_final)
            stitched = self._stitcher.process(event)
            if not stitched:
                return
            self._transcript_buffer.append(
                source=self._translated_channel,
                text=stitched.text,
                is_final=stitched.is_final,
            )
            if stitched.should_speak:
                try:
                    self._tts_queue.put_nowait(stitched.text)
                except Exception:
                    if self._on_error:
                        self._on_error(f"{self._translated_channel}: tts queue overflow")
        except Exception as exc:
            if self._on_error:
                self._on_error(f"{self._translated_channel}: {exc}")

    def _start_tts_worker(self) -> None:
        self._tts_stop.clear()
        self._tts_thread = Thread(target=self._run_tts_worker, daemon=True)
        self._tts_thread.start()

    def _stop_tts_worker(self) -> None:
        self._tts_stop.set()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=1.0)
        self._tts_thread = None
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except Empty:
                break

    def _run_tts_worker(self) -> None:
        while not self._tts_stop.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                audio = self._tts.synthesize(text, sample_rate=self._tts_sample_rate)
                self._audio_playback.play(
                    audio=audio,
                    sample_rate=self._tts_sample_rate,
                    output_device_name=self._get_output_device(),
                )
            except Exception as exc:
                if self._on_error:
                    self._on_error(f"tts playback failed: {exc}")

    def stats(self) -> dict[str, object]:
        capture = self._audio_capture.stats()
        asr = self._asr_stream.stats()
        return {
            "running": self._running,
            "capture_running": capture.running,
            "capture_rate": capture.sample_rate,
            "capture_frames": capture.frame_count,
            "capture_level": capture.level,
            "capture_error": capture.last_error,
            "asr_queue": asr.queue_size,
            "asr_dropped": asr.dropped_chunks,
            "asr_partials": asr.partial_count,
            "asr_finals": asr.final_count,
            "asr_last": asr.last_debug,
            "vad_rms": asr.vad_rms,
            "vad_threshold": asr.vad_threshold,
        }
