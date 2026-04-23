from __future__ import annotations

import unittest

from app.infra.asr.text_correction import AsrTextCorrector, _looks_like_safe_cjk_surface_fix
from app.infra.config.schema import LlmConfig
from app.infra.translation.lm_studio_adapter import LmStudioClient


class _StubCorrectionClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def correct_asr_text(self, *, text: str, language: str, context=None) -> str:
        return self._response


class AsrTextCorrectionTests(unittest.TestCase):
    def test_extract_correction_rejects_structured_reply_junk(self) -> None:
        cleaned = LmStudioClient._extract_correction_text('{"translation":"text"}')

        self.assertEqual(cleaned, "")

    def test_corrector_rejects_large_chinese_rewrite(self) -> None:
        corrector = AsrTextCorrector(LlmConfig(), enabled=True)
        corrector._client = _StubCorrectionClient("這是一段過度改寫而且還加入額外資訊的句子")

        result = corrector.correct("這句話原本很短", language="zh-TW")

        self.assertFalse(result.applied)
        self.assertEqual(result.text, "這句話原本很短")

    def test_corrector_accepts_small_chinese_fix(self) -> None:
        corrector = AsrTextCorrector(LlmConfig(), enabled=True)
        corrector._client = _StubCorrectionClient("裝腔作勢")

        result = corrector.correct("撞槍做事", language="zh-TW")

        self.assertTrue(result.applied)
        self.assertEqual(result.text, "裝腔作勢")

    def test_safe_cjk_surface_fix_accepts_same_length_phrase(self) -> None:
        self.assertTrue(_looks_like_safe_cjk_surface_fix("撞槍做事", "裝腔作勢"))

    def test_safe_cjk_surface_fix_rejects_mixed_script_rewrite(self) -> None:
        self.assertFalse(_looks_like_safe_cjk_surface_fix("撞槍做事", "please forgive me"))


if __name__ == "__main__":
    unittest.main()
