from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from app.infra.asr.backend_v2 import BackendDescriptor, BackendTranscript
from app.infra.asr.endpointing_v2 import build_endpointing_runtime
from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.config.schema import AppConfig, VadSettings


class AsrV2EndpointingTests(unittest.TestCase):
    class _FakeBackend:
        def __init__(self, name: str, is_final: bool) -> None:
            self.descriptor = BackendDescriptor(name=name, mode="test", streaming=not is_final)
            self._is_final = is_final

        def transcribe_partial(self, audio, sample_rate: int) -> BackendTranscript:
            return BackendTranscript(text="partial text", is_final=False, detected_language="zh")

        def transcribe_final(self, audio, sample_rate: int) -> BackendTranscript:
            return BackendTranscript(text="final text", is_final=True, detected_language="zh")

    def test_rms_endpointing_runtime_detects_speech_start_and_hard_endpoint(self) -> None:
        vad = VadSettings(
            backend="rms",
            min_speech_duration_ms=80,
            min_silence_duration_ms=240,
            speech_pad_ms=80,
            rms_threshold=0.02,
        )
        runtime = build_endpointing_runtime("rms", vad)
        speech = np.full((1600,), 0.05, dtype=np.float32)
        silence = np.zeros((3200,), dtype=np.float32)

        started = runtime.update(speech, 16000)
        paused = runtime.update(silence, 16000)
        ended = runtime.update(silence, 16000)

        self.assertTrue(started.speech_started)
        self.assertTrue(paused.soft_endpoint)
        self.assertTrue(ended.hard_endpoint)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["speech_started_count"], 1)
        self.assertGreaterEqual(int(snapshot["soft_endpoint_count"]), 1)
        self.assertGreaterEqual(int(snapshot["hard_endpoint_count"]), 1)

    @patch("app.infra.asr.endpointing_v2._SileroStreamingVad")
    def test_neural_endpointing_runtime_uses_probability_threshold(self, mock_vad_cls) -> None:
        backend = mock_vad_cls.return_value
        backend.available = True
        backend.probability.return_value = 0.91

        vad = VadSettings(
            backend="silero_vad",
            min_speech_duration_ms=80,
            min_silence_duration_ms=240,
            speech_pad_ms=80,
            neural_threshold=0.5,
            rms_threshold=0.02,
        )
        runtime = build_endpointing_runtime("neural_endpoint", vad)
        decision = runtime.update(np.zeros((1600,), dtype=np.float32), 16000)

        self.assertTrue(decision.speech_started)
        self.assertTrue(decision.speech_active)
        snapshot = runtime.snapshot()
        self.assertTrue(snapshot["available"])
        self.assertGreaterEqual(float(snapshot["speech_probability"]), 0.9)

    def test_legacy_fsmn_backend_alias_maps_to_silero(self) -> None:
        vad = VadSettings(
            backend="fsmn_vad",
            min_speech_duration_ms=80,
            min_silence_duration_ms=240,
            speech_pad_ms=80,
            neural_threshold=0.5,
            rms_threshold=0.02,
        )
        runtime = build_endpointing_runtime("fsmn_vad", vad)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["backend"], "silero_vad")

    @patch("app.infra.asr.manager_v2.build_backend_pair")
    def test_manager_v2_reports_shadow_endpointing_stats(self, mock_build_backend_pair) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"
        cfg.runtime.asr_v2_endpointing = "rms"
        cfg.asr_channels.local.vad.backend = "rms"
        cfg.asr_channels.remote.vad.backend = "rms"
        cfg.asr_channels.local.vad.min_speech_duration_ms = 80
        cfg.asr_channels.local.vad.min_silence_duration_ms = 240
        cfg.asr_channels.local.vad.speech_pad_ms = 80
        cfg.asr_channels.remote.vad.min_speech_duration_ms = 80
        cfg.asr_channels.remote.vad.min_silence_duration_ms = 240
        cfg.asr_channels.remote.vad.speech_pad_ms = 80

        mock_build_backend_pair.return_value = (
            self._FakeBackend("fake:partial", is_final=False),
            self._FakeBackend("fake:final", is_final=True),
        )
        manager = ASRManagerV2(cfg, pipeline_revision=7)
        manager.start("local", lambda _event: None)
        speech = np.full((1600,), 0.05, dtype=np.float32)
        silence = np.zeros((3200,), dtype=np.float32)

        manager.submit("local", speech, 16000)
        time.sleep(0.06)
        manager.submit("local", silence, 16000)
        time.sleep(0.06)
        manager.submit("local", silence, 16000)
        time.sleep(0.18)

        stats = manager.stats()["local"]

        self.assertEqual(stats["pipeline_mode"], "v2")
        self.assertEqual(stats["execution_mode"], "native_v2")
        self.assertEqual(stats["endpointing"]["backend"], "rms")
        self.assertGreaterEqual(int(stats["endpointing"]["speech_started_count"]), 1)
        self.assertGreaterEqual(
            int(stats["endpointing"]["soft_endpoint_count"]) + int(stats["endpointing"]["hard_endpoint_count"]),
            1,
        )
        self.assertGreaterEqual(int(stats["final_count"]), 1)

    @patch("app.infra.asr.manager_v2.build_backend_pair")
    def test_manager_v2_emits_partial_and_final_events(self, mock_build_backend_pair) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"
        cfg.runtime.asr_v2_endpointing = "rms"
        cfg.asr_channels.local.vad.backend = "rms"
        cfg.asr_channels.local.vad.min_speech_duration_ms = 80
        cfg.asr_channels.local.vad.min_silence_duration_ms = 240
        cfg.asr_channels.local.vad.speech_pad_ms = 80
        cfg.asr_channels.local.streaming.partial_interval_ms = 200
        mock_build_backend_pair.return_value = (
            self._FakeBackend("fake:partial", is_final=False),
            self._FakeBackend("fake:final", is_final=True),
        )
        events = []
        manager = ASRManagerV2(cfg, pipeline_revision=9)
        manager.start("local", events.append)

        speech = np.full((16000,), 0.05, dtype=np.float32)
        silence = np.zeros((3200,), dtype=np.float32)
        manager.submit("local", speech, 16000)
        time.sleep(0.12)
        manager.submit("local", silence, 16000)
        manager.submit("local", silence, 16000)
        time.sleep(0.2)

        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(any(not event.is_final for event in events))
        self.assertTrue(any(event.is_final for event in events))
        self.assertEqual(events[-1].text, "final text")

    @patch("app.infra.asr.manager_v2.build_backend_pair")
    def test_manager_v2_finalizes_on_soft_endpoint_and_restarts_next_utterance(self, mock_build_backend_pair) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_pipeline = "v2"
        cfg.runtime.asr_v2_endpointing = "rms"
        cfg.runtime.asr_partial_min_audio_ms = 240
        cfg.asr_channels.local.vad.backend = "rms"
        cfg.asr_channels.local.vad.min_speech_duration_ms = 80
        cfg.asr_channels.local.vad.min_silence_duration_ms = 500
        cfg.asr_channels.local.vad.speech_pad_ms = 80
        cfg.asr_channels.local.streaming.partial_interval_ms = 200
        cfg.asr_channels.local.streaming.soft_final_audio_ms = 7200
        mock_build_backend_pair.return_value = (
            self._FakeBackend("fake:partial", is_final=False),
            self._FakeBackend("fake:final", is_final=True),
        )
        events = []
        manager = ASRManagerV2(cfg, pipeline_revision=10)
        manager.start("local", events.append)

        speech = np.full((16000,), 0.05, dtype=np.float32)
        soft_pause = np.zeros((4800,), dtype=np.float32)

        manager.submit("local", speech, 16000)
        time.sleep(0.08)
        manager.submit("local", soft_pause, 16000)
        time.sleep(0.18)
        manager.submit("local", speech, 16000)
        time.sleep(0.08)
        manager.submit("local", soft_pause, 16000)
        time.sleep(0.24)

        finals = [event for event in events if event.is_final]
        self.assertGreaterEqual(len(finals), 2)
        self.assertTrue(all(event.is_early_final for event in finals))


if __name__ == "__main__":
    unittest.main()
