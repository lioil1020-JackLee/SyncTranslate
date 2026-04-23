from __future__ import annotations

from dataclasses import dataclass
from threading import Event
import time
import unittest

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.asr.contracts import ASREventWithSource
from app.infra.config.schema import AudioRouteConfig


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


class _CorrectingTranslatorManager(_FakeTranslatorManager):
    def correct_asr_event(self, event):
        return ASREventWithSource(
            source=event.source,
            utterance_id=event.utterance_id,
            revision=event.revision,
            pipeline_revision=event.pipeline_revision,
            config_fingerprint=event.config_fingerprint,
            created_at=event.created_at,
            text="corrected text",
            is_final=event.is_final,
            is_early_final=event.is_early_final,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            latency_ms=event.latency_ms,
            detected_language=event.detected_language,
            raw_text=event.text,
            correction_applied=True,
        )


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
    def test_router_starts_only_sources_with_asr_enabled(self) -> None:
        router, input_manager, asr_manager, translator, _ = _build_router(enabled_by_source={"remote": True, "local": True})
        config = type("Cfg", (), {"runtime": type("Rt", (), {"remote_asr_language": "en", "local_asr_language": "none"})()})()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("remote", input_manager.start_calls)
        self.assertIn("remote", asr_manager.start_calls)
        self.assertNotIn("local", input_manager.start_calls)
        self.assertNotIn("local", asr_manager.start_calls)

    def test_passthrough_toggle_changes_capture_need_for_local_source(self) -> None:
        router, input_manager, asr_manager, translator, _ = _build_router(enabled_by_source={"remote": True, "local": True})
        config = type("Cfg", (), {"runtime": type("Rt", (), {"remote_asr_language": "en", "local_asr_language": "none"})()})()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        router.set_output_mode("remote", "passthrough")
        self.assertIn("local", input_manager.start_calls)
        self.assertFalse(asr_manager.enabled["local"])

        router.set_output_mode("remote", "tts")
        self.assertIn("local", input_manager.stop_calls)

    def test_subtitle_only_mode_does_not_trigger_passthrough_capture(self) -> None:
        router, input_manager, asr_manager, translator, _ = _build_router(enabled_by_source={"remote": True, "local": True})
        config = type("Cfg", (), {"runtime": type("Rt", (), {"remote_asr_language": "en", "local_asr_language": "none"})()})()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("meeting_to_local", routes, sample_rate=24000, chunk_ms=40)

        self.assertNotIn("local", input_manager.start_calls)
        router.set_output_mode("remote", "subtitle_only")
        self.assertFalse(asr_manager.enabled["local"])

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

    def test_original_panel_keeps_raw_asr_when_correction_is_applied(self) -> None:
        translator = _CorrectingTranslatorManager(enabled=False)
        router, _, _, _, _ = _build_router(translation_enabled=False, translator_manager=translator)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-raw",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="raw asr text",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=50,
            detected_language="zh-TW",
        )

        router._on_asr_event(event)

        original = router._transcript_buffer.latest("meeting_original", limit=1)
        translated = router._transcript_buffer.latest("meeting_translated", limit=1)
        self.assertEqual(original[0].text, "raw asr text")
        self.assertEqual(translated[0].text, "corrected text")

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
        router, input_manager, asr_manager, translator, _ = _build_router(enabled_by_source={"remote": True, "local": True})
        config = type("Cfg", (), {"runtime": type("Rt", (), {"remote_asr_language": "en", "local_asr_language": "none"})()})()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")
        router.start("unknown_mode", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("remote", input_manager.start_calls)
        self.assertIn("remote", asr_manager.start_calls)
        self.assertNotIn("local", input_manager.start_calls)
        self.assertNotIn("local", asr_manager.start_calls)

    def test_local_to_meeting_mode_starts_only_local_asr_path(self) -> None:
        router, input_manager, asr_manager, _, _ = _build_router(enabled_by_source={"remote": True, "local": True})
        config = type("Cfg", (), {"runtime": type("Rt", (), {"remote_asr_language": "en", "local_asr_language": "zh-TW"})()})()
        router.refresh_runtime_config(config)  # type: ignore[arg-type]
        routes = AudioRouteConfig(meeting_in="remote-dev", microphone_in="local-dev", speaker_out="spk", meeting_out="mtg")

        router.start("local_to_meeting", routes, sample_rate=24000, chunk_ms=40)

        self.assertIn("local", input_manager.start_calls)
        self.assertIn("local", asr_manager.start_calls)
        self.assertNotIn("remote", input_manager.start_calls)
        self.assertNotIn("remote", asr_manager.start_calls)

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


class AudioRouterRuntimeConfigTests(unittest.TestCase):
    """Tests that verify the runtime config snapshot and policy helpers work correctly."""

    def test_refresh_runtime_config_stores_snapshot_on_router(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.display_partial_strategy = "all"
        router.refresh_runtime_config(config)
        self.assertIs(router._runtime_config, config)

    def test_display_partial_strategy_reads_from_runtime_snapshot(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.display_partial_strategy = "all"
        router.refresh_runtime_config(config)
        self.assertEqual(router._display_partial_strategy(), "all")

    def test_display_partial_strategy_defaults_to_stable_only_without_config(self) -> None:
        router, _, _, _, _ = _build_router()
        self.assertEqual(router._display_partial_strategy(), "stable_only")

    def test_stable_partial_min_repeats_reads_from_runtime_snapshot(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.stable_partial_min_repeats = 5
        router.refresh_runtime_config(config)
        self.assertEqual(router._stable_partial_min_repeats(), 5)

    def test_stable_partial_progression_accepts_large_append_when_prefix_is_stable(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.partial_stability_max_delta_chars = 6
        router.refresh_runtime_config(config)

        previous = "We need to update the deployment pipeline"
        current = "We need to update the deployment pipeline before Friday morning"

        self.assertTrue(router._is_stable_partial_progression(previous, current))

    def test_stable_partial_progression_rejects_mid_sentence_rewrite(self) -> None:
        router, _, _, _, _ = _build_router()
        previous = "We need to update the deployment pipeline"
        current = "We should replace the release checklist"

        self.assertFalse(router._is_stable_partial_progression(previous, current))

    def test_put_drop_oldest_returns_false_when_queue_not_full(self) -> None:
        from queue import Queue
        router, _, _, _, _ = _build_router()
        q = Queue(maxsize=4)
        dropped = router._put_drop_oldest(q, "item-1")
        self.assertFalse(dropped)
        self.assertEqual(q.qsize(), 1)

    def test_put_drop_oldest_returns_true_and_drops_oldest_when_full(self) -> None:
        from queue import Queue
        router, _, _, _, _ = _build_router()
        q = Queue(maxsize=2)
        q.put_nowait("old-1")
        q.put_nowait("old-2")
        dropped = router._put_drop_oldest(q, "new-item")
        self.assertTrue(dropped)
        self.assertEqual(q.qsize(), 2)
        items = [q.get_nowait(), q.get_nowait()]
        self.assertIn("new-item", items)

    def test_should_drop_over_latency_drops_when_latency_exceeds_max(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.max_pipeline_latency_ms = 1000
        router.refresh_runtime_config(config)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-slow",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="slow",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=2000,
            latency_ms=1500,  # exceeds 1000ms max
            detected_language="en",
        )
        self.assertTrue(router._should_drop_over_latency(event))

    def test_should_drop_over_latency_keeps_when_within_limit(self) -> None:
        router, _, _, _, _ = _build_router()
        from app.infra.config.schema import AppConfig
        config = AppConfig()
        config.runtime.max_pipeline_latency_ms = 3000
        router.refresh_runtime_config(config)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-fast",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="fast",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=200,
            detected_language="en",
        )
        self.assertFalse(router._should_drop_over_latency(event))

    def test_stats_includes_translation_overflow_field(self) -> None:
        router, _, _, _, _ = _build_router()
        stats = router.stats()
        self.assertIn("translation_overflow", stats.__dataclass_fields__)
        self.assertEqual(stats.translation_overflow, {"local": 0, "remote": 0})

    def test_translation_queue_overflow_increments_counter_and_emits_diagnostic(self) -> None:
        router, _, _, _, _ = _build_router(async_translation=True)
        from queue import Queue
        diagnostics: list[str] = []
        router._on_diagnostic_event = diagnostics.append  # type: ignore[method-assign]
        # Fill the queue to capacity then trigger overflow
        q = router._translation_queues["remote"]
        while not q.full():
            q.put_nowait(object())
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-overflow",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="overflow text",
            is_final=False,
            is_early_final=False,
            start_ms=0,
            end_ms=100,
            latency_ms=50,
            detected_language="en",
        )
        router._enqueue_translation_event(event)
        self.assertEqual(router._translation_overflow_counters["remote"], 1)
        self.assertTrue(any("translation_queue_overflow" in d for d in diagnostics))

    def test_stop_resets_translation_overflow_counters(self) -> None:
        router, _, _, _, _ = _build_router()
        router._translation_overflow_counters["remote"] = 5
        router.stop()
        self.assertEqual(router._translation_overflow_counters, {"local": 0, "remote": 0})


if __name__ == "__main__":
    unittest.main()
