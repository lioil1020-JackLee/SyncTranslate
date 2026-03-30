from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.config.schema import AppConfig
from app.ui.pages.live_caption_page import LiveCaptionPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class LiveCaptionLabelTests(_QtTestCase):
    def test_manually_selected_asr_language_updates_original_panel_labels(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "ja"
        config.runtime.local_asr_language = "th"

        page.apply_config(config)

        self.assertIn("手動：日文", page.remote_original_label.text())
        self.assertIn("手動：泰文", page.local_original_label.text())

    def test_tts_mode_keeps_translated_panel_labels(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "ja"
        config.runtime.local_translation_target = "ko"
        config.runtime.remote_tts_voice = "ja-JP-NanamiNeural"
        config.runtime.local_tts_voice = "ko-KR-SunHiNeural"

        page.apply_config(config)
        page.update_translation_voice_labels(config)

        self.assertEqual(page.remote_translated_label.text(), "遠端翻譯")
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")

    def test_voice_choice_alone_no_longer_switches_panel_into_translation_mode(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "none"
        config.runtime.remote_tts_voice = "ja-JP-NanamiNeural"
        config.runtime.local_tts_voice = "th-TH-PremwadeeNeural"

        page.apply_config(config)
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))
        page.local_tts_voice_combo.setCurrentIndex(page.local_tts_voice_combo.findData("none"))
        page.update_translation_voice_labels(config)

        self.assertEqual(page.remote_translated_label.text(), "遠端輸出")
        self.assertEqual(page.local_translated_label.text(), "本地輸出")

    def test_passthrough_mode_survives_config_round_trip(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "zh-TW"
        config.runtime.remote_tts_voice = "zh-TW-HsiaoChenNeural"
        config.runtime.remote_translation_enabled = True
        config.runtime.remote_tts_enabled = True

        page.apply_config(config)
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        updated = AppConfig()
        page.update_config(updated)

        reloaded = LiveCaptionPage()
        reloaded.apply_config(updated)

        self.assertEqual(updated.runtime.remote_translation_enabled, True)
        self.assertEqual(updated.runtime.remote_tts_enabled, False)
        self.assertEqual(reloaded.selected_tts_output_mode_for_channel("remote"), "passthrough")


if __name__ == "__main__":
    unittest.main()
