from __future__ import annotations

import unittest
import numpy as np

from app.infra.asr.streaming_pipeline import ASRManager
from app.infra.config.schema import AppConfig
from app.infra.asr.stream_worker import StreamingAsr, _looks_like_silence_hallucination
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

        self.assertEqual(local_provider._client.model, "local-model")
        self.assertEqual(remote_provider._client.model, "remote-model")

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
        self.assertEqual(translator_manager._providers["local"]._client.model, "local-model")
        self.assertEqual(translator_manager._providers["remote"]._client.model, "remote-model")

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

    def test_passthrough_gain_boosts_streaming_audio(self) -> None:
        cfg = AppConfig()
        cfg.runtime.passthrough_gain = 2.0
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
        self.assertAlmostEqual(float(forwarded[0, 0]), 0.5, places=5)

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
            _looks_like_silence_hallucination("晚安", audio_ms=1200, vad_rms=0.01)
        )
        self.assertFalse(
            _looks_like_silence_hallucination("Can you hear me now?", audio_ms=1200, vad_rms=0.04)
        )

    def test_streaming_asr_keeps_segment_helpers_on_class(self) -> None:
        self.assertTrue(hasattr(StreamingAsr, "_segment_audio_ms"))
        self.assertTrue(hasattr(StreamingAsr, "_limited_audio"))
        self.assertTrue(hasattr(StreamingAsr, "_reset_segment"))
        self.assertTrue(hasattr(StreamingAsr, "_append_pre_roll_chunk"))
        self.assertTrue(hasattr(StreamingAsr, "_prime_segment_from_pre_roll"))

    def test_translation_cleanup_rejects_overexpanded_zh_output(self) -> None:
        cleaned = LmStudioClient._clean_translation_output(
            "從現在開始，我會成為你的朋友。我也愛你，我能感受到你的存在。",
            target_lang="zh-TW",
        )
        self.assertEqual(cleaned, "")


if __name__ == "__main__":
    unittest.main()
