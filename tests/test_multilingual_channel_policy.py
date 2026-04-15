from __future__ import annotations

import unittest
import numpy as np
import sys
from types import ModuleType
from unittest.mock import patch

from app.infra.asr.streaming_pipeline import ASRManager
from app.infra.asr.language_policy import VadSegmenter
from app.infra.config.schema import AppConfig
from app.infra.asr.stream_worker import (
    StreamingAsr,
    _looks_like_known_non_speech_text,
    _looks_like_script_mismatch_junk,
    _looks_like_silence_hallucination,
    _transcript_drop_reason,
)
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine, _clear_model_cache_for_tests
from app.infra.translation.engine import TranslatorManager
from app.infra.translation.lm_studio_adapter import LmStudioClient
from app.infra.tts.playback_queue import TTSManager
from app.infra.tts.voice_policy import resolve_edge_voice_for_target


class _DummyPlayback:
    def __init__(self) -> None:
        self.passthrough_calls: list[tuple[np.ndarray, float, str]] = []

    def play(self, audio, sample_rate, output_device_name, *, blocking: bool = False) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_volume(self, volume: float) -> None:
        pass

    def push_passthrough(self, *, audio, sample_rate: float, output_device_name: str) -> None:
        self.passthrough_calls.append((audio, sample_rate, output_device_name))


class MultiLingualChannelPolicyTests(unittest.TestCase):
    def tearDown(self) -> None:
        _clear_model_cache_for_tests()
        super().tearDown()

    def test_asr_profile_and_queue_always_follow_source_channel(self) -> None:
        cfg = AppConfig()
        # Swap away from previous zh/en assumptions.
        cfg.language.local_source = "en"
        cfg.language.meeting_source = "ja"
        cfg.asr_channels.local.model = "local-asr-model"
        cfg.asr_channels.remote.model = "remote-asr-model"
        cfg.runtime.asr_queue_maxsize_local = 25
        cfg.runtime.asr_queue_maxsize_remote = 31
        manager = ASRManager(cfg)

        self.assertEqual(manager._asr_profile_for_source("local").model, "local-asr-model")
        self.assertEqual(manager._asr_profile_for_source("remote").model, "remote-asr-model")
        self.assertEqual(manager._asr_queue_maxsize_for_source("local"), 25)
        self.assertEqual(manager._asr_queue_maxsize_for_source("remote"), 31)

    def test_translation_provider_always_follows_source_channel(self) -> None:
        cfg = AppConfig()
        # Non zh/en directions: provider selection should still be stable by source channel.
        cfg.language.local_source = "ja"
        cfg.language.local_target = "de"
        cfg.language.meeting_source = "fr"
        cfg.language.meeting_target = "es"
        cfg.llm_channels.local.model = "local-model"
        cfg.llm_channels.remote.model = "remote-model"
        manager = TranslatorManager(cfg)
        local_provider = manager._providers["local"]
        remote_provider = manager._providers["remote"]

        self.assertEqual(local_provider._client.model, "hy-mt1.5-7b")
        self.assertEqual(remote_provider._client.model, "hy-mt1.5-7b")

    def test_shared_models_do_not_override_direction_specific_selection(self) -> None:
        cfg = AppConfig()
        cfg.asr.model = "shared-asr-model"
        cfg.asr_channels.local.model = "local-asr-model"
        cfg.asr_channels.remote.model = "remote-asr-model"
        cfg.llm.model = "shared-llm-model"
        cfg.llm_channels.local.model = "local-model"
        cfg.llm_channels.remote.model = "remote-model"

        asr_manager = ASRManager(cfg)
        translator_manager = TranslatorManager(cfg)

        self.assertEqual(asr_manager._asr_profile_for_source("local").model, "local-asr-model")
        self.assertEqual(asr_manager._asr_profile_for_source("remote").model, "remote-asr-model")
        self.assertEqual(translator_manager._providers["local"]._client.model, "hy-mt1.5-7b")
        self.assertEqual(translator_manager._providers["remote"]._client.model, "hy-mt1.5-7b")

    def test_asr_auto_mode_does_not_pin_language(self) -> None:
        cfg = AppConfig()
        cfg.language.local_source = "ja"
        cfg.language.meeting_source = "th"
        cfg.runtime.asr_language_mode = "auto"
        manager = ASRManager(cfg)

        self.assertEqual(manager._asr_language_for_source("local"), "")
        self.assertEqual(manager._asr_language_for_source("remote"), "")

    def test_tts_voice_fallback_covers_ja_ko_th(self) -> None:
        cfg = AppConfig()
        cfg.meeting_tts.voice_name = "zh-TW-HsiaoChenNeural"
        cfg.local_tts.voice_name = "en-US-JennyNeural"

        self.assertEqual(resolve_edge_voice_for_target(cfg, "ja"), "ja-JP-NanamiNeural")
        self.assertEqual(resolve_edge_voice_for_target(cfg, "ko"), "ko-KR-SunHiNeural")
        self.assertEqual(resolve_edge_voice_for_target(cfg, "th"), "th-TH-PremwadeeNeural")

    def test_tts_channel_config_uses_source_side_ui_selection_after_channel_crossing(self) -> None:
        cfg = AppConfig()
        cfg.language.meeting_target = "en"
        cfg.language.local_target = "ko"
        cfg.runtime.remote_tts_voice = "en-US-JennyNeural"
        cfg.runtime.local_tts_voice = "ko-KR-InJoonNeural"
        manager = TTSManager(
            config=cfg,
            local_playback=_DummyPlayback(),
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )

        local_channel_cfg = manager._channel_tts_config("local")
        remote_channel_cfg = manager._channel_tts_config("remote")

        self.assertEqual(local_channel_cfg.voice_name, "en-US-JennyNeural")
        self.assertEqual(remote_channel_cfg.voice_name, "ko-KR-InJoonNeural")

    def test_passthrough_downmix_keeps_right_channel_signal(self) -> None:
        cfg = AppConfig()
        local_playback = _DummyPlayback()
        manager = TTSManager(
            config=cfg,
            local_playback=local_playback,
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "passthrough")
        manager._passthrough_warmup_until["local"] = 0.0

        stereo = np.zeros((4, 2), dtype=np.float32)
        stereo[:, 1] = 0.5
        manager.submit_passthrough("local", stereo, 48000.0)

        forwarded, sample_rate, output_device = local_playback.passthrough_calls[0]
        self.assertEqual(forwarded.shape, (4, 2))
        self.assertGreater(float(np.max(np.abs(forwarded[:, 1]))), 0.0)
        self.assertEqual(sample_rate, 48000.0)
        self.assertEqual(output_device, "speaker")

    def test_passthrough_uses_raw_streaming_audio_before_playback_volume(self) -> None:
        cfg = AppConfig()
        local_playback = _DummyPlayback()
        manager = TTSManager(
            config=cfg,
            local_playback=local_playback,
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "passthrough")
        manager._passthrough_warmup_until["local"] = 0.0

        mono = np.full((4, 1), 0.25, dtype=np.float32)
        manager.submit_passthrough("local", mono, 48000.0)

        forwarded, _, _ = local_playback.passthrough_calls[0]
        self.assertAlmostEqual(float(forwarded[0, 0]), 0.25, places=5)

    def test_asr_submit_prefers_strongest_input_channel(self) -> None:
        cfg = AppConfig()
        manager = ASRManager(cfg)

        class _Stream:
            def __init__(self) -> None:
                self.last_chunk = None
                self.last_rate = None

            def submit_chunk(self, chunk, sample_rate):
                self.last_chunk = chunk
                self.last_rate = sample_rate

            def stop(self) -> None:
                pass

        stream = _Stream()
        manager._enabled["remote"] = True
        manager._stream_of = lambda source: stream  # type: ignore[method-assign]

        stereo = np.zeros((4, 2), dtype=np.float32)
        stereo[:, 1] = 0.4
        manager.submit("remote", stereo, 48000.0)

        self.assertIsNotNone(stream.last_chunk)
        self.assertEqual(stream.last_chunk.shape, (4,))
        self.assertAlmostEqual(float(stream.last_chunk[0]), 0.4, places=5)

    def test_asr_submit_averages_similar_stereo_channels_for_stable_mono(self) -> None:
        cfg = AppConfig()
        manager = ASRManager(cfg)

        class _Stream:
            def __init__(self) -> None:
                self.chunks: list[np.ndarray] = []

            def submit_chunk(self, chunk, sample_rate):
                self.chunks.append(chunk)

            def stop(self) -> None:
                pass

        stream = _Stream()
        manager._enabled["remote"] = True
        manager._stream_of = lambda source: stream  # type: ignore[method-assign]

        stereo = np.array(
            [
                [0.30, 0.20],
                [0.26, 0.16],
                [0.22, 0.12],
                [0.18, 0.08],
            ],
            dtype=np.float32,
        )

        manager.submit("remote", stereo, 48000.0)

        self.assertEqual(len(stream.chunks), 1)
        self.assertAlmostEqual(float(stream.chunks[0][0]), 0.25, places=5)

    def test_faster_whisper_engines_with_same_runtime_share_model_and_lock(self) -> None:
        class _FakeWhisperModel:
            init_count = 0

            def __init__(self, model: str, device: str, compute_type: str) -> None:
                type(self).init_count += 1
                self.model = model
                self.device = device
                self.compute_type = compute_type

            def transcribe(self, *_args, **_kwargs):
                return [], {"language": "en"}

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
        original = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = fake_module
        try:
            first = FasterWhisperEngine(model="large-v3", device="cuda", compute_type="float16")
            second = FasterWhisperEngine(model="large-v3", device="cuda", compute_type="float16")

            model_a = first._get_model()
            model_b = second._get_model()

            self.assertIs(model_a, model_b)
            self.assertIs(first._transcribe_lock, second._transcribe_lock)
            self.assertEqual(_FakeWhisperModel.init_count, 1)
        finally:
            if original is None:
                sys.modules.pop("faster_whisper", None)
            else:
                sys.modules["faster_whisper"] = original

    def test_asr_submit_avoids_phase_cancellation_from_stereo_sum(self) -> None:
        cfg = AppConfig()
        manager = ASRManager(cfg)

        class _Stream:
            def __init__(self) -> None:
                self.last_chunk = None

            def submit_chunk(self, chunk, sample_rate):
                self.last_chunk = chunk

            def stop(self) -> None:
                pass

        stream = _Stream()
        manager._enabled["remote"] = True
        manager._stream_of = lambda source: stream  # type: ignore[method-assign]

        stereo = np.zeros((4, 2), dtype=np.float32)
        stereo[:, 0] = 0.35
        stereo[:, 1] = -0.35
        manager.submit("remote", stereo, 48000.0)

        self.assertIsNotNone(stream.last_chunk)
        self.assertEqual(stream.last_chunk.shape, (4,))
        self.assertGreater(float(np.max(np.abs(stream.last_chunk))), 0.3)

    def test_short_low_energy_thank_you_like_text_is_treated_as_hallucination(self) -> None:
        self.assertTrue(
            _looks_like_silence_hallucination("Thank you all.", audio_ms=900, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("\u8b1d\u8b1d\u5927\u5bb6", audio_ms=900, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("by bwd6", audio_ms=900, vad_rms=0.0)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("晚安", audio_ms=1200, vad_rms=0.01)
        )
        self.assertFalse(
            _looks_like_silence_hallucination("Can you hear me now?", audio_ms=1200, vad_rms=0.04)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("Thank you for watching.", audio_ms=1200, vad_rms=0.01)
        )
        self.assertTrue(
            _looks_like_silence_hallucination("感謝您的收看", audio_ms=1200, vad_rms=0.01)
        )

    def test_known_non_speech_overlay_lines_are_filtered(self) -> None:
        self.assertTrue(
            _looks_like_known_non_speech_text("優優獨播劇場——YoYo Television Series Exclusive")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("字幕由 Amara.org 社群提供")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("請不吝點贊 訂閱 轉發 打賞支援明鏡與點點欄目")
        )
        self.assertTrue(
            _looks_like_known_non_speech_text("MING PAO CANADA MING PAO TORONTO")
        )
        self.assertFalse(
            _looks_like_known_non_speech_text("你趕快修煉我來處理船上的黑衣樓殺手")
        )

    def test_short_foreign_script_junk_is_filtered_when_language_is_pinned(self) -> None:
        self.assertTrue(
            _looks_like_script_mismatch_junk("Ак", expected_language="zh")
        )
        self.assertTrue(
            _looks_like_script_mismatch_junk("Ак", expected_language="en")
        )
        self.assertFalse(
            _looks_like_script_mismatch_junk("你好", expected_language="zh")
        )
        self.assertFalse(
            _looks_like_script_mismatch_junk("OK", expected_language="en")
        )

    def test_transcript_drop_reason_prioritizes_overlay_and_script_junk(self) -> None:
        self.assertEqual(
            _transcript_drop_reason(
                "謝謝觀看,下次見!",
                audio_ms=2400,
                vad_rms=0.08,
                expected_language="zh",
            ),
            "non-speech-overlay",
        )
        self.assertEqual(
            _transcript_drop_reason(
                "Ак",
                audio_ms=900,
                vad_rms=0.01,
                expected_language="zh",
            ),
            "script-mismatch",
        )
        self.assertEqual(
            _transcript_drop_reason(
                "by bwd6",
                audio_ms=900,
                vad_rms=0.0,
                expected_language="zh",
            ),
            "hallucinated",
        )

    def test_effective_pre_roll_ms_keeps_enough_lead_in_for_soft_sentence_starts(self) -> None:
        self.assertEqual(
            ASRManager._effective_pre_roll_ms(
                configured_pre_roll_ms=220,
                min_speech_duration_ms=150,
                speech_pad_ms=320,
            ),
            310,
        )
        self.assertEqual(
            ASRManager._effective_pre_roll_ms(
                configured_pre_roll_ms=480,
                min_speech_duration_ms=150,
                speech_pad_ms=320,
            ),
            480,
        )

    def test_streaming_asr_keeps_segment_helpers_on_class(self) -> None:
        self.assertTrue(hasattr(StreamingAsr, "_segment_audio_ms"))
        self.assertTrue(hasattr(StreamingAsr, "_limited_audio"))
        self.assertTrue(hasattr(StreamingAsr, "_reset_segment"))
        self.assertTrue(hasattr(StreamingAsr, "_append_pre_roll_chunk"))
        self.assertTrue(hasattr(StreamingAsr, "_prime_segment_from_pre_roll"))

    def test_streaming_asr_uses_runtime_partial_floor_instead_of_old_hard_clamp(self) -> None:
        class _DummyEngine:
            pass

        class _DummyVad:
            last_rms = 0.0
            effective_rms_threshold = 0.0
            _config = type("_Cfg", (), {"min_silence_duration_ms": 220})()

            def reset(self) -> None:
                pass

        stream = StreamingAsr(
            engine=_DummyEngine(),  # type: ignore[arg-type]
            vad=_DummyVad(),  # type: ignore[arg-type]
            partial_interval_ms=250,
            partial_interval_floor_ms=250,
            min_partial_audio_ms=240,
        )

        self.assertEqual(stream._partial_interval_ms, 250)
        self.assertEqual(stream._min_partial_audio_ms, 240)

    def test_streaming_asr_adapts_for_short_turns(self) -> None:
        stream = StreamingAsr(
            engine=object(),  # type: ignore[arg-type]
            vad=VadSegmenter(type("Cfg", (), {
                "enabled": True,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 520,
                "max_speech_duration_s": 10.0,
                "speech_pad_ms": 320,
                "rms_threshold": 0.02,
            })()),  # type: ignore[arg-type]
            partial_interval_ms=800,
            partial_interval_floor_ms=520,
            soft_final_audio_ms=4200,
            adaptive_enabled=True,
        )

        for idx in range(4):
            stream._record_final_adaptation(audio_ms=1400, latency_ms=420, now_ms=1000 + idx)

        stats = stream.stats()
        self.assertEqual(stats.adaptive_mode, "short_turn")
        self.assertLess(stats.adaptive_partial_interval_ms, 800)
        self.assertLess(stats.adaptive_min_silence_duration_ms, 520)
        self.assertLess(stats.adaptive_soft_final_audio_ms, 4200)

    def test_streaming_asr_load_sheds_when_recent_latency_is_high(self) -> None:
        stream = StreamingAsr(
            engine=object(),  # type: ignore[arg-type]
            vad=VadSegmenter(type("Cfg", (), {
                "enabled": True,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 520,
                "max_speech_duration_s": 10.0,
                "speech_pad_ms": 320,
                "rms_threshold": 0.02,
            })()),  # type: ignore[arg-type]
            partial_interval_ms=800,
            partial_interval_floor_ms=520,
            soft_final_audio_ms=4200,
            adaptive_enabled=True,
        )

        for idx in range(4):
            stream._record_partial_adaptation(latency_ms=1200 + idx * 20)

        stats = stream.stats()
        self.assertIn("load_shed", stats.adaptive_mode)
        self.assertGreater(stats.adaptive_partial_interval_ms, 800)
        self.assertLess(stats.adaptive_soft_final_audio_ms, 4200)

    def test_vad_reports_short_pause_before_full_finalize(self) -> None:
        vad = VadSegmenter(type("Cfg", (), {
            "enabled": True,
            "min_speech_duration_ms": 80,
            "min_silence_duration_ms": 900,
            "max_speech_duration_s": 10.0,
            "speech_pad_ms": 520,
            "rms_threshold": 0.02,
        })())

        speech = np.full((1600,), 0.05, dtype=np.float32)
        silence = np.zeros((3200,), dtype=np.float32)

        vad.update(speech, 16000)
        active = vad.update(speech, 16000)
        paused = vad.update(silence, 16000)

        self.assertTrue(active.speech_active)
        self.assertGreater(paused.pause_ms, 0.0)
        self.assertFalse(paused.finalize)

    def test_streaming_asr_soft_split_prefers_pause_over_fixed_length(self) -> None:
        class _DummyVad:
            last_rms = 0.0
            effective_rms_threshold = 0.02
            effective_min_silence_duration_ms = 900

            def reset(self) -> None:
                pass

            def set_runtime_tuning(self, *, min_silence_duration_ms: int | None = None) -> None:
                pass

        stream = StreamingAsr(
            engine=object(),  # type: ignore[arg-type]
            vad=_DummyVad(),  # type: ignore[arg-type]
            soft_final_audio_ms=4200,
            adaptive_enabled=False,
        )
        stream._segment_sample_rate = 16000
        stream._segment_chunks = [np.zeros((16000 * 5,), dtype=np.float32)]

        self.assertFalse(stream._should_emit_soft_split(type("Decision", (), {"pause_ms": 0.0})()))
        self.assertTrue(stream._should_emit_soft_split(type("Decision", (), {"pause_ms": 240.0})()))

    def test_speaker_diarization_is_disabled_by_default(self) -> None:
        cfg = AppConfig()
        manager = ASRManager(cfg)

        self.assertIsNone(manager._speaker_diarizer_for_source("local"))

    @patch("app.infra.asr.language_policy._SileroStreamingVad")
    def test_neural_vad_backend_can_drive_speech_detection(self, mock_vad_cls) -> None:
        backend = mock_vad_cls.return_value
        backend.available = True
        backend.probability.return_value = 0.92

        vad = VadSegmenter(type("Cfg", (), {
            "enabled": True,
            "backend": "silero_vad",
            "min_speech_duration_ms": 80,
            "min_silence_duration_ms": 900,
            "max_speech_duration_s": 10.0,
            "speech_pad_ms": 520,
            "rms_threshold": 0.02,
            "neural_threshold": 0.5,
        })())

        decision = vad.update(np.zeros((1600,), dtype=np.float32), 16000)

        self.assertTrue(decision.speech_active)
        backend.probability.assert_called()

    def test_tts_queue_can_accept_stable_partial_before_final(self) -> None:
        cfg = AppConfig()
        cfg.runtime.tts_accept_stable_partial = True
        cfg.runtime.tts_partial_min_chars = 4
        manager = TTSManager(
            config=cfg,
            local_playback=_DummyPlayback(),
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "tts")

        manager.enqueue(
            "local",
            "stable partial text",
            utterance_id="u1",
            revision=1,
            is_final=False,
            is_stable_partial=True,
        )

        self.assertEqual(len(manager._pending), 1)
        self.assertTrue(manager._pending[0].is_stable_partial)

    def test_tts_new_final_keeps_other_utterances_in_queue(self) -> None:
        cfg = AppConfig()
        manager = TTSManager(
            config=cfg,
            local_playback=_DummyPlayback(),
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "tts")

        manager.enqueue("local", "first sentence", utterance_id="u1", revision=1, is_final=True)
        manager.enqueue("local", "second sentence", utterance_id="u2", revision=1, is_final=True)

        self.assertEqual([task.utterance_id for task in manager._pending], ["u1", "u2"])

    def test_tts_new_revision_replaces_only_same_utterance(self) -> None:
        cfg = AppConfig()
        manager = TTSManager(
            config=cfg,
            local_playback=_DummyPlayback(),
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "tts")

        manager.enqueue("local", "old one", utterance_id="u1", revision=1, is_final=True)
        manager.enqueue("local", "other one", utterance_id="u2", revision=1, is_final=True)
        manager.enqueue("local", "new one", utterance_id="u1", revision=2, is_final=True)

        self.assertEqual(
            [(task.utterance_id, task.text, task.revision) for task in manager._pending],
            [("u2", "other one", 1), ("u1", "new one", 2)],
        )

    def test_final_tts_task_is_not_dropped_by_wait_timeout(self) -> None:
        cfg = AppConfig()
        cfg.runtime.tts_max_wait_ms = 1
        manager = TTSManager(
            config=cfg,
            local_playback=_DummyPlayback(),
            remote_playback=_DummyPlayback(),
            get_local_output_device=lambda: "speaker",
            get_remote_output_device=lambda: "meeting",
        )
        manager.set_output_mode("local", "tts")
        manager.enqueue("local", "final sentence", utterance_id="u1", revision=1, is_final=True)
        manager._pending[0].created_at -= 10.0

        task = manager._next_text_task("local")

        self.assertIsNotNone(task)
        self.assertEqual(task.text, "final sentence")

    def test_translation_cleanup_rejects_overexpanded_zh_output(self) -> None:
        cleaned = LmStudioClient._clean_translation_output(
            "從現在開始，我會成為你的朋友。我也愛你，我能感受到你的存在。",
            target_lang="zh-TW",
        )
        self.assertEqual(cleaned, "")


    def test_short_zh_translation_is_still_displayable(self) -> None:
        from app.infra.translation.stitcher import _looks_like_displayable_zh_translation

        self.assertTrue(_looks_like_displayable_zh_translation("不。"))
        self.assertTrue(_looks_like_displayable_zh_translation("是。"))


if __name__ == "__main__":
    unittest.main()
