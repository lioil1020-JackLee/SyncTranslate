from __future__ import annotations

from dataclasses import dataclass
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.asr.contracts import ASREventWithSource
from app.infra.config.schema import AudioRouteConfig
from app.infra.translation.engine import TranslationEvent
from app.ui.main_window import MainWindow
from app.ui.pages.live_caption_page import LiveCaptionPage


@dataclass
class _CaptureStats:
    running: bool
    sample_rate: float
    frame_count: int
    level: float
    last_error: str


class _FakeInputManager:
    def __init__(self) -> None:
        self.running = {"local": False, "remote": False}
        self._consumers: dict[str, list] = {"local": [], "remote": []}

    def start(self, source: str, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        self.running[source] = True

    def stop(self, source: str) -> None:
        self.running[source] = False

    def stop_all(self) -> None:
        self.running = {"local": False, "remote": False}

    def add_consumer(self, source: str, consumer) -> None:
        self._consumers[source].append(consumer)

    def remove_consumer(self, source: str, consumer) -> None:
        self._consumers[source] = [item for item in self._consumers[source] if item is not consumer]

    def emit(self, source: str, chunk: np.ndarray, sample_rate: float) -> None:
        if not self.running[source]:
            return
        for consumer in list(self._consumers[source]):
            consumer(chunk, sample_rate)

    def stats(self) -> dict[str, _CaptureStats]:
        return {
            "local": _CaptureStats(self.running["local"], 24000.0, 256 if self.running["local"] else 0, 0.1, ""),
            "remote": _CaptureStats(self.running["remote"], 24000.0, 256 if self.running["remote"] else 0, 0.1, ""),
        }


class _FakeAsrManager:
    def __init__(self) -> None:
        self.enabled = {"local": False, "remote": False}
        self._callbacks: dict[str, object] = {}
        self._submitted = {"local": 0, "remote": 0}

    def set_enabled(self, source: str, enabled: bool) -> None:
        self.enabled[source] = enabled

    def start(self, source: str, callback) -> None:
        self._callbacks[source] = callback

    def stop(self, source: str) -> None:
        self._callbacks.pop(source, None)

    def stop_all(self) -> None:
        self._callbacks.clear()

    def submit(self, source: str, chunk, sample_rate: float) -> None:
        if not self.enabled.get(source, False):
            return
        callback = self._callbacks.get(source)
        if callback is None:
            return
        self._submitted[source] += 1
        batch = self._build_events(source)
        for event in batch:
            callback(event)

    def stats(self) -> dict[str, dict[str, object]]:
        return {
            "local": {
                "queue_size": 0,
                "dropped_chunks": 0,
                "partial_count": 1 if self._submitted["local"] else 0,
                "final_count": 1 if self._submitted["local"] else 0,
                "vad_rms": 0.0,
                "vad_threshold": 0.0,
                "last_debug": "",
            },
            "remote": {
                "queue_size": 0,
                "dropped_chunks": 0,
                "partial_count": 2 if self._submitted["remote"] else 0,
                "final_count": 1 if self._submitted["remote"] else 0,
                "vad_rms": 0.0,
                "vad_threshold": 0.0,
                "last_debug": "",
            },
        }

    @staticmethod
    def _build_events(source: str) -> list[ASREventWithSource]:
        if source == "remote":
            return [
                ASREventWithSource(
                    source="remote",
                    utterance_id="remote-1",
                    revision=1,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=0.0,
                    text="hello",
                    is_final=False,
                    is_early_final=False,
                    start_ms=0,
                    end_ms=300,
                    latency_ms=40,
                    detected_language="en",
                ),
                ASREventWithSource(
                    source="remote",
                    utterance_id="remote-1",
                    revision=2,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=0.1,
                    text="hello world",
                    is_final=True,
                    is_early_final=False,
                    start_ms=0,
                    end_ms=900,
                    latency_ms=55,
                    detected_language="en",
                ),
                ASREventWithSource(
                    source="remote",
                    utterance_id="remote-2",
                    revision=1,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=0.2,
                    text="next sentence",
                    is_final=False,
                    is_early_final=False,
                    start_ms=1000,
                    end_ms=1400,
                    latency_ms=35,
                    detected_language="en",
                ),
                ASREventWithSource(
                    source="remote",
                    utterance_id="remote-2",
                    revision=2,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=0.3,
                    text="next sentence now",
                    is_final=False,
                    is_early_final=False,
                    start_ms=1000,
                    end_ms=1550,
                    latency_ms=38,
                    detected_language="en",
                ),
            ]
        return [
            ASREventWithSource(
                source="local",
                utterance_id="local-1",
                revision=1,
                pipeline_revision=1,
                config_fingerprint="fp",
                created_at=0.0,
                text="ni hao",
                is_final=False,
                is_early_final=False,
                start_ms=0,
                end_ms=250,
                latency_ms=30,
                detected_language="zh-TW",
            ),
            ASREventWithSource(
                source="local",
                utterance_id="local-1",
                revision=2,
                pipeline_revision=1,
                config_fingerprint="fp",
                created_at=0.1,
                text="ni hao ma",
                is_final=True,
                is_early_final=False,
                start_ms=0,
                end_ms=850,
                latency_ms=45,
                detected_language="zh-TW",
            ),
        ]


class _FakeTranslatorManager:
    def translation_enabled(self) -> bool:
        return True

    @staticmethod
    def original_channel_of(source: str) -> str:
        return "meeting_original" if source == "remote" else "local_original"

    def process(self, event: ASREventWithSource) -> TranslationEvent | None:
        if event.source == "remote":
            return TranslationEvent(
                source="remote",
                utterance_id=event.utterance_id,
                revision=event.revision,
                created_at=event.created_at,
                original_channel="meeting_original",
                translated_channel="meeting_translated",
                tts_channel="local",
                text=f"ZH:{event.text}",
                speak_text=f"SAY_LOCAL:{event.text}",
                is_final=event.is_final,
                is_stable_partial=not event.is_final,
                is_early_final=event.is_early_final,
                should_display=True,
                should_speak=event.is_final,
            )
        return TranslationEvent(
            source="local",
            utterance_id=event.utterance_id,
            revision=event.revision,
            created_at=event.created_at,
            original_channel="local_original",
            translated_channel="local_translated",
            tts_channel="remote",
            text=f"EN:{event.text}",
            speak_text=f"SAY_REMOTE:{event.text}",
            is_final=event.is_final,
            is_stable_partial=not event.is_final,
            is_early_final=event.is_early_final,
            should_display=True,
            should_speak=event.is_final,
        )


class _FakeTtsManager:
    def __init__(self) -> None:
        self._mode = {"local": "tts", "remote": "tts"}
        self.enqueued: list[tuple[str, str, bool]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_muted(self, channel: str, muted: bool) -> None:
        pass

    def set_passthrough_enabled(self, channel: str, enabled: bool) -> None:
        self._mode[channel] = "passthrough" if enabled else "tts"

    def set_output_mode(self, channel: str, mode: str) -> None:
        self._mode[channel] = mode

    def is_passthrough_enabled(self, channel: str) -> bool:
        return self._mode.get(channel) == "passthrough"

    def output_mode(self, channel: str) -> str:
        return self._mode.get(channel, "subtitle_only")

    def submit_passthrough(self, channel: str, chunk, sample_rate: float) -> None:
        pass

    def enqueue(
        self,
        channel: str,
        text: str,
        *,
        utterance_id: str,
        revision: int,
        is_final: bool,
        is_stable_partial: bool = False,
        is_early_final: bool = False,
    ) -> None:
        self.enqueued.append((channel, text, is_final))

    def stats(self) -> dict[str, object]:
        return {
            "queue_depth": len(self.enqueued),
            "queue_depth_local": sum(1 for item in self.enqueued if item[0] == "local"),
            "queue_depth_remote": sum(1 for item in self.enqueued if item[0] == "remote"),
            "drop_count_local": 0,
            "drop_count_remote": 0,
            "oldest_age_ms": 0.0,
        }


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class PipelineIntegrationTests(_QtTestCase):
    def test_fake_audio_flows_from_asr_to_translation_subtitles_and_tts(self) -> None:
        transcript_buffer = TranscriptBuffer()
        input_manager = _FakeInputManager()
        asr_manager = _FakeAsrManager()
        translator_manager = _FakeTranslatorManager()
        tts_manager = _FakeTtsManager()
        router = AudioRouter(
            transcript_buffer=transcript_buffer,
            input_manager=input_manager,  # type: ignore[arg-type]
            asr_manager=asr_manager,  # type: ignore[arg-type]
            translator_manager=translator_manager,  # type: ignore[arg-type]
            tts_manager=tts_manager,  # type: ignore[arg-type]
            state_manager=StateManager(local_echo_guard_enabled=True),
        )
        routes = AudioRouteConfig(
            meeting_in="remote-dev",
            microphone_in="local-dev",
            speaker_out="speaker-dev",
            meeting_out="meeting-dev",
        )
        router.start("bidirectional", routes, sample_rate=24000, chunk_ms=40)

        chunk = np.ones((240, 1), dtype=np.float32)
        input_manager.emit("remote", chunk, 24000.0)
        input_manager.emit("local", chunk, 24000.0)

        remote_original_items = transcript_buffer.latest("meeting_original", limit=10)
        remote_translated_items = transcript_buffer.latest("meeting_translated", limit=10)
        local_original_items = transcript_buffer.latest("local_original", limit=10)
        local_translated_items = transcript_buffer.latest("local_translated", limit=10)

        self.assertEqual([item.text for item in remote_original_items], ["hello world", "next sentence now"])
        self.assertEqual([item.is_final for item in remote_original_items], [True, False])
        self.assertEqual([item.text for item in remote_translated_items], ["ZH:hello world", "ZH:next sentence now"])
        self.assertEqual([item.is_final for item in remote_translated_items], [True, False])
        self.assertEqual([item.text for item in local_original_items], ["ni hao ma"])
        self.assertEqual([item.text for item in local_translated_items], ["EN:ni hao ma"])
        self.assertEqual(
            tts_manager.enqueued,
            [
                ("local", "SAY_LOCAL:hello world", True),
                ("remote", "SAY_REMOTE:ni hao ma", True),
            ],
        )

        page = LiveCaptionPage()
        page.set_remote_original_lines(MainWindow._build_transcript_lines(remote_original_items))
        page.set_remote_translated_lines(MainWindow._build_transcript_lines(remote_translated_items))
        page.set_local_original_lines(MainWindow._build_transcript_lines(local_original_items))
        page.set_local_translated_lines(MainWindow._build_transcript_lines(local_translated_items))

        self.assertEqual(
            page.remote_original.toPlainText(),
            "[partial] next sentence now\n[final] hello world",
        )
        self.assertEqual(
            page.remote_translated.toPlainText(),
            "[partial] ZH:next sentence now\n[final] ZH:hello world",
        )
        self.assertEqual(page.local_original.toPlainText(), "[final] ni hao ma")
        self.assertEqual(page.local_translated.toPlainText(), "[final] EN:ni hao ma")

    def test_unstable_partial_is_hidden_until_a_stable_follow_up_arrives(self) -> None:
        transcript_buffer = TranscriptBuffer()
        input_manager = _FakeInputManager()
        asr_manager = _FakeAsrManager()
        translator_manager = _FakeTranslatorManager()
        tts_manager = _FakeTtsManager()
        router = AudioRouter(
            transcript_buffer=transcript_buffer,
            input_manager=input_manager,  # type: ignore[arg-type]
            asr_manager=asr_manager,  # type: ignore[arg-type]
            translator_manager=translator_manager,  # type: ignore[arg-type]
            tts_manager=tts_manager,  # type: ignore[arg-type]
            state_manager=StateManager(local_echo_guard_enabled=True),
        )

        router._on_asr_event(  # type: ignore[attr-defined]
            ASREventWithSource(
                source="remote",
                utterance_id="u-1",
                revision=1,
                pipeline_revision=1,
                config_fingerprint="fp",
                created_at=0.0,
                text="draft sentence",
                is_final=False,
                is_early_final=False,
                start_ms=0,
                end_ms=400,
                latency_ms=20,
                detected_language="en",
            )
        )
        self.assertEqual(transcript_buffer.latest("meeting_original", limit=10), [])
        self.assertEqual(transcript_buffer.latest("meeting_translated", limit=10), [])

        router._on_asr_event(  # type: ignore[attr-defined]
            ASREventWithSource(
                source="remote",
                utterance_id="u-1",
                revision=2,
                pipeline_revision=1,
                config_fingerprint="fp",
                created_at=0.1,
                text="draft sentence now",
                is_final=False,
                is_early_final=False,
                start_ms=0,
                end_ms=700,
                latency_ms=25,
                detected_language="en",
            )
        )
        self.assertEqual([item.text for item in transcript_buffer.latest("meeting_original", limit=10)], ["draft sentence now"])
        self.assertEqual([item.text for item in transcript_buffer.latest("meeting_translated", limit=10)], ["ZH:draft sentence now"])


if __name__ == "__main__":
    unittest.main()
