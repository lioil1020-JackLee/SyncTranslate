from __future__ import annotations

import unittest

from app.infra.asr.text_correction import AsrTextCorrector
from app.infra.config.schema import LlmConfig
from app.infra.translation.lm_studio_adapter import LmStudioClient


class _StubCorrectionClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def correct_asr_text(self, *, text: str, language: str, context=None) -> str:
        return self._response


class AsrTextCorrectionTests(unittest.TestCase):
    def test_extract_correction_rejects_structured_reply_junk(self) -> None:
        cleaned = LmStudioClient._extract_correction_text('名字，他說要把桌子買下。」} json')

        self.assertEqual(cleaned, "")

    def test_corrector_rejects_large_chinese_rewrite(self) -> None:
        corrector = AsrTextCorrector(LlmConfig(), enabled=True)
        corrector._client = _StubCorrectionClient("另外還擺了幾顆火龍果，象徵著茶葉的民間風味。")

        result = corrector.correct("桌上放著青菜。", language="zh-TW")

        self.assertFalse(result.applied)
        self.assertEqual(result.text, "桌上放著青菜。")

    def test_corrector_accepts_small_chinese_fix(self) -> None:
        corrector = AsrTextCorrector(LlmConfig(), enabled=True)
        corrector._client = _StubCorrectionClient("拾荒")

        result = corrector.correct("石荒", language="zh-TW")

        self.assertTrue(result.applied)
        self.assertEqual(result.text, "拾荒")


if __name__ == "__main__":
    unittest.main()
