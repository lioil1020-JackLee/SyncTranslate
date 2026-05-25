from __future__ import annotations

import unittest

from app.infra.asr.pipeline_v2 import resolve_requested_asr_language
from app.infra.config.schema import AppConfig


class AsrLanguageResolutionTests(unittest.TestCase):
    def test_auto_mode_compatibility_resolves_to_fixed_product_defaults(self) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_language_mode = "auto"

        self.assertEqual(resolve_requested_asr_language(cfg, "local"), "zh-TW")
        self.assertEqual(resolve_requested_asr_language(cfg, "remote"), "en")

    def test_auto_mode_still_respects_explicit_none_override(self) -> None:
        cfg = AppConfig()
        cfg.runtime.asr_language_mode = "auto"
        cfg.runtime.local_asr_language = "none"

        self.assertEqual(resolve_requested_asr_language(cfg, "local"), "none")


if __name__ == "__main__":
    unittest.main()
