from __future__ import annotations

import unittest

from app.infra.asr.streaming_pipeline import ASRManager
from app.infra.config.schema import AppConfig
from app.infra.translation.engine import TranslatorManager
from app.infra.tts.voice_policy import resolve_edge_voice_for_target


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


if __name__ == "__main__":
    unittest.main()
