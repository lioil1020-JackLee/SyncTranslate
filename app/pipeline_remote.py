from __future__ import annotations

from typing import Callable

import numpy as np

from app.asr_worker import AsrResult, AsrWorker
from app.audio_playback import AudioPlayback
from app.audio_capture import AudioCapture
from app.model_providers import AsrProvider, TranslateProvider, TtsProvider
from app.transcript_buffer import TranscriptBuffer


class RemotePipeline:
    def __init__(
        self,
        audio_capture: AudioCapture,
        transcript_buffer: TranscriptBuffer,
        audio_playback: AudioPlayback,
        asr_provider: AsrProvider,
        translate_provider: TranslateProvider,
        tts_provider: TtsProvider,
        get_local_tts_output_device: Callable[[], str],
        on_original_transcript: Callable[[str], None],
        on_translated_transcript: Callable[[str], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._audio_capture = audio_capture
        self._transcript_buffer = transcript_buffer
        self._audio_playback = audio_playback
        self._get_local_tts_output_device = get_local_tts_output_device
        self._on_original_transcript = on_original_transcript
        self._on_translated_transcript = on_translated_transcript
        self._on_error = on_error
        self._asr_worker = AsrWorker(asr_provider=asr_provider)
        self._translate_worker = translate_provider
        self._tts_worker = tts_provider
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, input_device_name: str, sample_rate: int) -> None:
        self.stop()
        self._asr_worker.start(self._handle_asr_result)
        self._audio_capture.add_consumer(self._on_audio_chunk)
        self._audio_capture.start(input_device_name, sample_rate=sample_rate)
        self._running = True

    def stop(self) -> None:
        self._audio_capture.remove_consumer(self._on_audio_chunk)
        self._audio_capture.stop()
        self._audio_playback.stop()
        self._asr_worker.stop()
        self._running = False

    def _on_audio_chunk(self, chunk: np.ndarray, sample_rate: float) -> None:
        self._asr_worker.submit_chunk(chunk, sample_rate)

    def _handle_asr_result(self, result: AsrResult) -> None:
        try:
            self._transcript_buffer.append(source="remote_original", text=result.text, is_final=result.is_final)
            translated = self._translate_worker.translate(result.text)
            self._transcript_buffer.append(source="remote_translated", text=translated, is_final=result.is_final)

            output_device = self._get_local_tts_output_device()
            audio = self._tts_worker.synthesize(translated)
            self._audio_playback.play(audio=audio, sample_rate=24000, output_device_name=output_device)

            self._on_original_transcript(result.text)
            self._on_translated_transcript(translated)
        except Exception as exc:
            if self._on_error:
                self._on_error(f"remote_pipeline error: {exc}")
