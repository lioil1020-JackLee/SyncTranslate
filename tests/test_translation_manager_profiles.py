from __future__ import annotations

import unittest
from unittest.mock import patch

from app.infra.asr.streaming_pipeline import ASREventWithSource
from app.infra.config.schema import AppConfig
from app.infra.translation.engine import TranslatorManager


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        context=None,
        profile=None,
    ) -> str:
        profile_name = getattr(profile, "name", "default")
        self.calls.append((text, source_lang, target_lang, profile_name))
        if profile_name == "speech_output_natural":
            return f"spoken:{text}"
        return f"caption:{text}"

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def capabilities(self):
        return None


class TranslatorManagerProfileTests(unittest.TestCase):
    @patch("app.infra.translation.engine.create_translation_provider")
    def test_speech_profile_can_generate_different_spoken_text(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.llm.caption_profile = "live_caption_fast"
        config.llm.speech_profile = "speech_output_natural"
        config.language.meeting_target = "en"

        manager = TranslatorManager(config)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u1",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="hello world",
            is_final=True,
            start_ms=0,
            end_ms=1200,
            latency_ms=80,
            detected_language="en",
        )

        translated = manager.process(event)

        self.assertIsNotNone(translated)
        assert translated is not None
        self.assertEqual(translated.text, "caption:hello world")
        self.assertEqual(translated.speak_text, "spoken:hello world")

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_matching_caption_and_speech_profiles_reuse_caption_text(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.llm.caption_profile = "live_caption_fast"
        config.llm.speech_profile = "live_caption_fast"

        manager = TranslatorManager(config)
        event = ASREventWithSource(
            source="local",
            utterance_id="u2",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="你好",
            is_final=True,
            start_ms=0,
            end_ms=1200,
            latency_ms=80,
            detected_language="zh-TW",
        )

        translated = manager.process(event)

        self.assertIsNotNone(translated)
        assert translated is not None
        self.assertEqual(translated.text, "caption:你好")
        self.assertEqual(translated.speak_text, "caption:你好")
        self.assertEqual(len(providers[0].calls), 1)


if __name__ == "__main__":
    unittest.main()
