from __future__ import annotations

import unittest

import numpy as np

from app.infra.asr.lexical_bias_v2 import AsrLexicalBiaser
from app.infra.asr.transcript_validator_v2 import AsrTranscriptValidatorV2


class AsrPostprocessV2Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
