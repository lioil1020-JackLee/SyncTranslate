from __future__ import annotations

from dataclasses import dataclass
from threading import Event
import time
import unittest

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.config.schema import AudioRouteConfig
from app.infra.asr.streaming_pipeline import ASREventWithSource


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
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []

    def start(self, source: str, device_name: str, sample_rate: int, chunk_ms: int) -> None:
        self.running[source] = True
        self.start_calls.append(source)

    def stop(self, source: str) -> None:
        self.running[source] = False
        self.stop_calls.append(source)

    def stop_all(self) -> None:
        self.running["local"] = False
        self.running["remote"] = False

    def add_consumer(self, source: str, consumer) -> None:
        pass

    def remove_consumer(self, source: str, consumer) -> None:
        pass

    def stats(self) -> dict[str, _CaptureStats]:
        return {
            "local": _CaptureStats(self.running["local"], 24000.0, 0, 0.0, ""),
            "remote": _CaptureStats(self.running["remote"], 24000.0, 0, 0.0, ""),
        }


class _FakeAsrManager:
    def __init__(self) -> None:
        self.enabled = {"local": False, "remote": False}
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.submit_calls: list[tuple[str, float]] = []
        self.configure_calls: list[tuple[object, int]] = []

    def set_enabled(self, source: str, enabled: bool) -> None:
        self.enabled[source] = enabled

    def start(self, source: str, callback) -> None:
        self.start_calls.append(source)

    def stop(self, source: str) -> None:
        self.stop_calls.append(source)

    def stop_all(self) -> None:
        pass

    def configure_pipeline(self, config, pipeline_revision: int) -> None:
        self.configure_calls.append((config, pipeline_revision))

    def submit(self, source: str, chunk, sample_rate: float) -> None:
        self.submit_calls.append((source, sample_rate))

    def stats(self) -> dict[str, dict[str, object]]:
        return {
            "local": {
                "queue_size": 0,
                "dropped_chunks": 0,
                "partial_count": 0,
                "final_count": 0,
                "vad_rms": 0.0,
                "vad_threshold": 0.0,
                "last_debug": "",
            },
            "remote": {
                "queue_size": 0,
                "dropped_chunks": 0,
                "partial_count": 0,
                "final_count": 0,
                "vad_rms": 0.0,
                "vad_threshold": 0.0,
                "last_debug": "",
            },
        }


class _FakeTranslatorManager:
    def __init__(self, *, enabled: bool = True, enabled_by_source: dict[str, bool] | None = None) -> None:
        self.enabled = enabled
        self.enabled_by_source = enabled_by_source or {}
        self.process_calls = 0

    def original_channel_of(self, source: str) -> str:
        return source

    def translation_enabled(self, source: str | None = None) -> bool:
        if source in self.enabled_by_source:
            return bool(self.enabled_by_source[source])
        return self.enabled

    def process(self, event):
        self.process_calls += 1
        return None


class _FakeTtsManager:
    def __init__(self) -> None:
        self._mode = {"local": "subtitle_only", "remote": "subtitle_only"}
        self.enqueued: list[tuple[str, str]] = []
        self.passthrough_calls: list[tuple[str, float]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_muted(self, channel: str, muted: bool) -> None:
        pass

    def set_passthrough_enabled(self, channel: str, enabled: bool) -> None:
        self._mode[channel] = "passthrough" if enabled else "subtitle_only"

    def set_output_mode(self, channel: str, mode: str) -> None:
        self._mode[channel] = mode

    def is_passthrough_enabled(self, channel: str) -> bool:
        return self._mode.get(channel) == "passthrough"

    def output_mode(self, channel: str) -> str:
        return self._mode.get(channel, "subtitle_only")

    def submit_passthrough(self, channel: str, chunk, sample_rate: float) -> None:
        self.passthrough_calls.append((channel, sample_rate))

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
        self.enqueued.append((channel, text))

    def stats(self) -> dict[str, object]:
        return {
            "queue_depth": 0,
            "queue_depth_local": 0,
            "queue_depth_remote": 0,
            "drop_count_local": 0,
            "drop_count_remote": 0,
            "oldest_age_ms": 0.0,
        }


def _build_router(
    *,
    translation_enabled: bool = True,
    enabled_by_source: dict[str, bool] | None = None,
    async_translation: bool = False,
    translator_manager=None,
) -> tuple[AudioRouter, _FakeInputManager, _FakeAsrManager, _FakeTranslatorManager, _FakeTtsManager]:
    input_manager = _FakeInputManager()
    asr_manager = _FakeAsrManager()
    translator_manager = translator_manager or _FakeTranslatorManager(
        enabled=translation_enabled,
        enabled_by_source=enabled_by_source,
    )
    tts_manager = _FakeTtsManager()
    router = AudioRouter(
        transcript_buffer=TranscriptBuffer(),
        input_manager=input_manager,  # type: ignore[arg-type]
        asr_manager=asr_manager,  # type: ignore[arg-type]
        translator_manager=translator_manager,  # type: ignore[arg-type]
        tts_manager=tts_manager,  # type: ignore[arg-type]
        state_manager=StateManager(local_echo_guard_enabled=True),
        async_translation=async_translation,
    )
    return router, input_manager, asr_manager, translator_manager, tts_manager


class AudioRouterPolicyTests(unittest.TestCase):
    def test_router_always_starts_both_sources(self) -> None:
        router, input_manager, asr_manager, _, _ = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("remote", input_manager.start_calls)
        self.assertIn("local", input_manager.start_calls)
        self.assertIn("remote", asr_manager.start_calls)
        self.assertIn("local", asr_manager.start_calls)

    def test_passthrough_toggle_changes_capture_need_for_local_source(self) -> None:
        router, input_manager, asr_manager, _, _ = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        router.set_output_mode("remote", "passthrough")
        self.assertIn("local", input_manager.start_calls)
        self.assertTrue(asr_manager.enabled["local"])

        router.set_output_mode("remote", "tts")
        self.assertNotIn("local", input_manager.stop_calls)

    def test_subtitle_only_mode_does_not_trigger_passthrough_capture(self) -> None:
        router, input_manager, asr_manager, _, _ = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("local", input_manager.start_calls)
        router.set_output_mode("remote", "subtitle_only")
        self.assertTrue(asr_manager.enabled["local"])

    def test_refresh_runtime_config_reconfigures_asr_pipeline(self) -> None:
        router, _, asr_manager, _, _ = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("bidirectional", routes, sample_rate=24000, chunk_ms=40)

        config = object()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]

        self.assertEqual(asr_manager.configure_calls[-1], (config, 1))

    def test_translation_disabled_skips_llm_and_updates_translated_panel(self) -> None:
        router, _, _, translator, tts = _build_router(translation_enabled=False)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u1",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="hello from asr",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="en",
        )
        router._on_asr_event(event)

        translated = router._transcript_buffer.latest("meeting_translated", limit=1)
        self.assertEqual(translator.process_calls, 0)
        self.assertEqual(len(translated), 1)
        self.assertEqual(translated[0].text, "hello from asr")
        self.assertEqual(tts.enqueued, [])

    def test_translation_disabled_with_tts_mode_speaks_asr_text(self) -> None:
        router, _, _, _, tts = _build_router(translation_enabled=False)
        router.set_output_mode("local", "tts")
        event = ASREventWithSource(
            source="remote",
            utterance_id="u2",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="final asr text",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="en",
        )
        router._on_asr_event(event)
        self.assertIn(("local", "final asr text"), tts.enqueued)

    def test_remote_translation_can_be_disabled_without_affecting_local_translation(self) -> None:
        router, _, _, translator, _ = _build_router(enabled_by_source={"remote": False, "local": True})
        remote_event = ASREventWithSource(
            source="remote",
            utterance_id="u3",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="remote asr only",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="en",
        )
        local_event = ASREventWithSource(
            source="local",
            utterance_id="u4",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="local translated",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="zh-TW",
        )

        router._on_asr_event(remote_event)
        router._on_asr_event(local_event)

        self.assertEqual(translator.process_calls, 1)
        remote_translated = router._transcript_buffer.latest("meeting_translated", limit=1)
        self.assertEqual(remote_translated[0].text, "remote asr only")

    def test_unknown_mode_is_ignored_and_router_stays_bidirectional(self) -> None:
        router, input_manager, asr_manager, _, _ = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("unknown_mode", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("local", input_manager.start_calls)
        self.assertIn("remote", input_manager.start_calls)
        self.assertIn("local", asr_manager.start_calls)
        self.assertIn("remote", asr_manager.start_calls)

    def test_local_source_passthrough_targets_remote_output_channel(self) -> None:
        router, _, _, _, tts = _build_router()
        router.set_output_mode("remote", "passthrough")

        router._on_local_audio_chunk(chunk=[0.0], sample_rate=16000.0)  # type: ignore[arg-type]

        self.assertIn(("remote", 16000.0), tts.passthrough_calls)

    def test_remote_source_passthrough_targets_local_output_channel(self) -> None:
        router, _, _, _, tts = _build_router()
        router.set_output_mode("local", "passthrough")

        router._on_remote_audio_chunk(chunk=[0.0], sample_rate=24000.0)  # type: ignore[arg-type]

        self.assertIn(("local", 24000.0), tts.passthrough_calls)

    def test_local_and_remote_chunk_handlers_follow_same_passthrough_then_asr_pattern(self) -> None:
        router, _, asr_manager, _, tts = _build_router()
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("bidirectional", routes, sample_rate=24000, chunk_ms=40)
        router._on_local_audio_chunk(chunk=[0.0], sample_rate=16000.0)  # type: ignore[arg-type]
        router._on_remote_audio_chunk(chunk=[0.0], sample_rate=24000.0)  # type: ignore[arg-type]

        self.assertIn(("remote", 16000.0), tts.passthrough_calls)
        self.assertIn(("local", 24000.0), tts.passthrough_calls)
        self.assertIn(("local", 16000.0), asr_manager.submit_calls)
        self.assertIn(("remote", 24000.0), asr_manager.submit_calls)

    def test_async_translation_keeps_asr_callback_non_blocking_per_direction(self) -> None:
        class _BlockingTranslator(_FakeTranslatorManager):
            def __init__(self) -> None:
                super().__init__(enabled=True)
                self.release = Event()

            def process(self, event):
                self.process_calls += 1
                self.release.wait(timeout=2.0)
                return None

        translator = _BlockingTranslator()
        router, _, _, _, _ = _build_router(async_translation=True, translator_manager=translator)
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("bidirectional", routes, sample_rate=24000, chunk_ms=40)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-block",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="hello",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="en",
        )

        started = time.perf_counter()
        router._on_asr_event(event)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        translator.release.set()
        router.stop()

        self.assertLess(elapsed_ms, 100.0)
        self.assertEqual(
            router._transcript_buffer.latest("meeting_original", limit=1)[0].text,
            "hello",
        )


if __name__ == "__main__":
    unittest.main()
