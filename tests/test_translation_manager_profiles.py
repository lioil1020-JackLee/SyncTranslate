from __future__ import annotations

import unittest
from unittest.mock import patch

from app.infra.asr.streaming_pipeline import ASREventWithSource
from app.infra.asr.text_correction import AsrCorrectionResult
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

    def debug_snapshot(self):
        return {"raw_response": "", "cleaned_response": "", "last_error": ""}


class _EmptyProvider(_FakeProvider):
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
        return ""

    def debug_snapshot(self):
        return {
            "raw_response": '{"translation":""}',
            "cleaned_response": "",
            "last_error": "",
        }


class _ContextSensitiveProvider(_FakeProvider):
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
        context_tuple = tuple(context or [])
        self.calls.append((text, source_lang, target_lang, profile_name))
        if context_tuple:
            return f"context:{len(context_tuple)}:{text}"
        return f"plain:{text}"


class _ScriptedProvider(_FakeProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)

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
        if self._responses:
            return self._responses.pop(0)
        return f"caption:{text}"


class TranslatorManagerProfileTests(unittest.TestCase):
    @patch("app.infra.translation.engine.AsrTextCorrector.correct")
    @patch("app.infra.translation.engine.create_translation_provider")
    def test_correct_asr_event_rewrites_final_text_for_downstream_use(self, mock_factory, mock_correct) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers
        mock_correct.return_value = AsrCorrectionResult(text="修正後文字", raw_text="修正前文字", applied=True)

        config = AppConfig()
        config.runtime.asr_final_correction_enabled = True
        manager = TranslatorManager(config)
        event = ASREventWithSource(
            source="local",
            utterance_id="u-correct",
            revision=2,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="修正前文字",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=900,
            latency_ms=50,
            detected_language="zh-TW",
        )

        corrected = manager.correct_asr_event(event)

        self.assertEqual(corrected.text, "修正後文字")
        self.assertEqual(corrected.raw_text, "修正前文字")
        self.assertTrue(corrected.correction_applied)

    @patch("app.infra.translation.engine.AsrTextCorrector.correct")
    @patch("app.infra.translation.engine.create_translation_provider")
    def test_correct_asr_event_leaves_partial_text_unchanged(self, mock_factory, mock_correct) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.runtime.asr_final_correction_enabled = True
        manager = TranslatorManager(config)
        event = ASREventWithSource(
            source="remote",
            utterance_id="u-partial",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="draft text",
            is_final=False,
            is_early_final=False,
            start_ms=0,
            end_ms=500,
            latency_ms=30,
            detected_language="en",
        )

        corrected = manager.correct_asr_event(event)

        self.assertIs(corrected, event)
        mock_correct.assert_not_called()

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_tts_speaks_caption_translation_text_directly(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.llm.caption_profile = "live_caption_fast"
        config.llm.speech_profile = "speech_output_natural"
        config.runtime.tts_use_speech_profile = True
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
            is_early_final=False,
            start_ms=0,
            end_ms=1200,
            latency_ms=80,
            detected_language="en",
        )

        translated = manager.process(event)

        self.assertIsNotNone(translated)
        assert translated is not None
        self.assertEqual(translated.text, "caption:hello world")
        self.assertEqual(translated.speak_text, "caption:hello world")
        self.assertEqual(sum(len(provider.calls) for provider in providers), 1)

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_matching_caption_and_speech_profiles_reuse_caption_text(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.llm.caption_profile = "live_caption_fast"
        config.llm.speech_profile = "live_caption_fast"
        config.runtime.tts_use_speech_profile = True

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
            is_early_final=False,
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

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_last_skip_reason_includes_provider_debug_when_translation_is_empty(self, mock_factory) -> None:
        providers = [_EmptyProvider(), _EmptyProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)
        event = ASREventWithSource(
            source="local",
            utterance_id="u3",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="你想睡覺喔",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=1500,
            latency_ms=60,
            detected_language="zh-TW",
        )

        translated = manager.process(event)

        self.assertIsNone(translated)
        reason = manager.last_skip_reason("local")
        self.assertIn("empty_translation", reason)
        self.assertIn("raw=", reason)

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_repeated_final_translation_still_marks_should_speak(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)
        event1 = ASREventWithSource(
            source="remote",
            utterance_id="u4",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="hello",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=1000,
            latency_ms=50,
            detected_language="en",
        )
        event2 = ASREventWithSource(
            source="remote",
            utterance_id="u5",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=1.0,
            text="hello",
            is_final=True,
            is_early_final=False,
            start_ms=1000,
            end_ms=2000,
            latency_ms=50,
            detected_language="en",
        )

        translated1 = manager.process(event1)
        translated2 = manager.process(event2)

        self.assertIsNotNone(translated1)
        self.assertIsNotNone(translated2)
        assert translated1 is not None
        assert translated2 is not None
        self.assertTrue(translated1.should_speak)
        self.assertTrue(translated2.should_speak)

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_final_translation_ignores_previous_context_for_same_text(self, mock_factory) -> None:
        providers = [_ContextSensitiveProvider(), _ContextSensitiveProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)
        first = ASREventWithSource(
            source="remote",
            utterance_id="u10",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="first sentence",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=1000,
            latency_ms=50,
            detected_language="en",
        )
        repeated = ASREventWithSource(
            source="remote",
            utterance_id="u11",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=1.0,
            text="same words",
            is_final=True,
            is_early_final=False,
            start_ms=1000,
            end_ms=2000,
            latency_ms=50,
            detected_language="en",
        )
        repeated_again = ASREventWithSource(
            source="remote",
            utterance_id="u12",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=2.0,
            text="same words",
            is_final=True,
            is_early_final=False,
            start_ms=2000,
            end_ms=3000,
            latency_ms=50,
            detected_language="en",
        )

        manager.process(first)
        translated1 = manager.process(repeated)
        translated2 = manager.process(repeated_again)

        self.assertIsNotNone(translated1)
        self.assertIsNotNone(translated2)
        assert translated1 is not None
        assert translated2 is not None
        self.assertEqual(translated1.text, "plain:same words")
        self.assertEqual(translated2.text, "plain:same words")

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_partial_translation_can_still_use_context_when_enabled(self, mock_factory) -> None:
        providers = [_ContextSensitiveProvider(), _ContextSensitiveProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.llm.sliding_window.enabled = True
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)

        first_final = ASREventWithSource(
            source="remote",
            utterance_id="u20",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=0.0,
            text="I saw Mary yesterday.",
            is_final=True,
            is_early_final=False,
            start_ms=0,
            end_ms=1200,
            latency_ms=50,
            detected_language="en",
        )
        followup_partial = ASREventWithSource(
            source="remote",
            utterance_id="u21",
            revision=1,
            pipeline_revision=1,
            config_fingerprint="fp",
            created_at=1.0,
            text="She said we should join the call right now",
            is_final=False,
            is_early_final=False,
            start_ms=1200,
            end_ms=3200,
            latency_ms=50,
            detected_language="en",
        )

        manager.process(first_final)
        translated = manager.process(followup_partial)

        self.assertIsNotNone(translated)
        assert translated is not None
        self.assertEqual(translated.text, "context:1:She said we should join the call right now")

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_llm_adapts_partial_gate_for_short_fragments(self, mock_factory) -> None:
        providers = [_FakeProvider(), _FakeProvider()]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)
        for idx in range(4):
            manager.process(
                ASREventWithSource(
                    source="remote",
                    utterance_id=f"short-{idx}",
                    revision=1,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=float(idx),
                    text="hi there",
                    is_final=True,
                    is_early_final=False,
                    start_ms=0,
                    end_ms=900,
                    latency_ms=50,
                    detected_language="en",
                )
            )

        snapshot = manager._stitchers["remote"].adaptive_snapshot()  # type: ignore[attr-defined]
        self.assertIn("short_fragments", str(snapshot["mode"]))
        self.assertGreater(int(snapshot["trigger_tokens"]), 16)
        self.assertGreater(int(snapshot["min_partial_interval_ms"]), 320)

    @patch("app.infra.translation.engine.create_translation_provider")
    def test_llm_switches_to_stable_profile_when_recent_failures_are_high(self, mock_factory) -> None:
        providers = [
            _ScriptedProvider(["caption:local"]),
            _ScriptedProvider(["<translation>noisy</translation>", "<translation>noisy</translation>", "caption:clean"]),
        ]
        mock_factory.side_effect = providers

        config = AppConfig()
        config.language.meeting_target = "en"
        manager = TranslatorManager(config)
        for idx in range(4):
            translated = manager.process(
                ASREventWithSource(
                    source="remote",
                    utterance_id=f"dirty-{idx}",
                    revision=1,
                    pipeline_revision=1,
                    config_fingerprint="fp",
                    created_at=float(idx),
                    text="this is a slightly longer sentence for adaptation",
                    is_final=True,
                    is_early_final=False,
                    start_ms=0,
                    end_ms=2200,
                    latency_ms=50,
                    detected_language="en",
                )
            )
            self.assertIsNone(translated)

        translated = manager.process(
            ASREventWithSource(
                source="remote",
                utterance_id="dirty-final",
                revision=1,
                pipeline_revision=1,
                config_fingerprint="fp",
                created_at=10.0,
                text="this is a slightly longer sentence for adaptation now",
                is_final=True,
                is_early_final=False,
                start_ms=0,
                end_ms=2200,
                latency_ms=50,
                detected_language="en",
            )
        )

        self.assertIsNotNone(translated)
        snapshot = manager._stitchers["remote"].adaptive_snapshot()  # type: ignore[attr-defined]
        self.assertEqual(snapshot["profile"], "live_caption_stable")
        self.assertEqual(providers[1].calls[-1][3], "live_caption_stable")


if __name__ == "__main__":
    unittest.main()
