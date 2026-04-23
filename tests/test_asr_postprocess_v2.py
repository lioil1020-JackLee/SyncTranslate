from __future__ import annotations

import unittest

import numpy as np

from app.infra.asr.backend_v2 import BackendDescriptor, BackendPostProcessor, FasterWhisperStreamingBackend
from app.infra.asr.faster_whisper_adapter import _is_hallucination
from app.infra.asr.lexical_bias_v2 import AsrLexicalBiaser
from app.infra.asr.transcript_validator_v2 import AsrTranscriptValidatorV2


class AsrPostprocessV2Tests(unittest.TestCase):
    class _FakeEngine:
        device = "cpu"

        def runtime_label(self) -> str:
            return "fake-engine"

    def test_lexical_bias_replaces_exact_alias(self) -> None:
        biaser = AsrLexicalBiaser("石荒=>拾荒\n白沙盾媽祖=>白沙屯媽祖", enabled=True)

        result = biaser.apply("白沙盾媽祖進香。", language="zh-TW")

        self.assertEqual(result, "白沙屯媽祖進香。")

    def test_lexical_bias_can_repair_near_match_for_cjk(self) -> None:
        biaser = AsrLexicalBiaser("西螺大橋=>西螺大橋", enabled=True)

        result = biaser.apply("西螺大僑封閉。", language="zh-TW")

        self.assertIn("西螺大橋", result)

    def test_validator_rejects_too_dense_text(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True, max_chars_per_second=8.0)
        audio = np.zeros((1600,), dtype=np.float32)

        result = validator.validate("這是一段非常長而且不合理地密集的文字內容", audio=audio, sample_rate=16000, language="zh-TW")

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "too-dense")

    def test_validator_rejects_looped_phrase(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True)
        audio = np.full((16000,), 0.02, dtype=np.float32)

        result = validator.validate("好好好好好好", audio=audio, sample_rate=16000, language="zh-TW")

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "looped-phrase")

    def test_validator_can_use_frontend_speech_ratio_hint(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True, min_speech_ratio_for_long_text=0.2)
        audio = np.zeros((32000,), dtype=np.float32)

        result = validator.validate(
            "這是一段應該保留下來的長句測試內容",
            audio=audio,
            sample_rate=16000,
            language="zh-TW",
            frontend_stats={"speech_ratio": 0.42},
        )

        self.assertTrue(result.accepted)

    def test_validator_keeps_non_cjk_long_text_with_low_speech_ratio(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True, min_speech_ratio_for_long_text=0.2)
        audio = np.zeros((32000,), dtype=np.float32)

        result = validator.validate(
            "this is a stable english sentence that should be kept",
            audio=audio,
            sample_rate=16000,
            language="en",
            frontend_stats={"speech_ratio": 0.02},
        )

        self.assertTrue(result.accepted)

    def test_validator_still_rejects_cjk_long_text_with_low_speech_ratio(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True, min_speech_ratio_for_long_text=0.2)
        audio = np.zeros((32000,), dtype=np.float32)

        result = validator.validate(
            "這是一段長句子但在低語音比例下應該被過濾避免幻覺內容",
            audio=audio,
            sample_rate=16000,
            language="zh-TW",
            frontend_stats={"speech_ratio": 0.02},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "low-speech-ratio")

    def test_validator_allows_cjk_final_with_borderline_low_speech_ratio(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True, min_speech_ratio_for_long_text=0.2)
        audio = np.zeros((32000,), dtype=np.float32)

        result = validator.validate(
            "這是一段已經完整辨識的中文句子內容",
            audio=audio,
            sample_rate=16000,
            language="zh-TW",
            frontend_stats={"speech_ratio": 0.1},
            is_final=True,
        )

        self.assertTrue(result.accepted)

    def test_validator_rejects_markup_leak_text(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True)
        audio = np.full((16000,), 0.02, dtype=np.float32)

        result = validator.validate(
            "所以就是到了親家那邊 </solution>",
            audio=audio,
            sample_rate=16000,
            language="zh-TW",
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "markup-leak")

    def test_validator_rejects_role_prefixed_llm_residue(self) -> None:
        validator = AsrTranscriptValidatorV2(enabled=True)
        audio = np.full((16000,), 0.02, dtype=np.float32)

        result = validator.validate(
            "assistant: this should not appear in subtitles",
            audio=audio,
            sample_rate=16000,
            language="en",
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "markup-leak")

    def test_hallucination_filter_keeps_english_sentence_on_english_channel(self) -> None:
        self.assertFalse(_is_hallucination("You can keep your safety.", language="en"))

    def test_hallucination_filter_can_still_block_latin_sentence_on_cjk_channel(self) -> None:
        self.assertTrue(_is_hallucination("You can keep your safety.", language="zh-TW"))

    def test_hallucination_filter_blocks_credit_and_service_overlay_phrases(self) -> None:
        self.assertTrue(_is_hallucination("字幕志願者 李宗盛", language="zh-TW"))
        self.assertTrue(_is_hallucination("服務員 買單", language="zh-TW"))

    def test_postprocessor_stats_track_rejection_reason(self) -> None:
        processor = BackendPostProcessor(
            language="zh-TW",
            biaser=AsrLexicalBiaser("", enabled=False),
            validator=AsrTranscriptValidatorV2(enabled=True),
        )
        audio = np.full((16000,), 0.02, dtype=np.float32)

        processor.process(
            "?隞亙停?臬鈭扛摰園??</solution>",
            audio=audio,
            sample_rate=16000,
        )

        stats = processor.stats()
        self.assertEqual(stats["rejected_count"], 1)
        self.assertEqual(stats["last_rejection_reason"], "markup-leak")
        self.assertEqual(stats["rejections_by_reason"]["markup-leak"], 1)

    def test_backend_runtime_info_exposes_postprocessor_stats(self) -> None:
        processor = BackendPostProcessor(
            language="zh-TW",
            biaser=AsrLexicalBiaser("", enabled=False),
            validator=AsrTranscriptValidatorV2(enabled=True),
        )
        audio = np.full((16000,), 0.02, dtype=np.float32)
        processor.process(
            "?隞亙停?臬鈭扛摰園??</solution>",
            audio=audio,
            sample_rate=16000,
        )
        backend = FasterWhisperStreamingBackend(
            engine=self._FakeEngine(),
            descriptor=BackendDescriptor(name="fake:partial", mode="test", streaming=True),
            post_processor=processor,
        )

        info = backend.runtime_info()

        self.assertIn("postprocessor", info)
        self.assertEqual(info["postprocessor"]["rejected_count"], 1)
        self.assertEqual(info["postprocessor"]["last_rejection_reason"], "markup-leak")


if __name__ == "__main__":
    unittest.main()
