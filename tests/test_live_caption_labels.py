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

    def test_tts_voice_labels_appear_on_translated_panels_when_tts_enabled(self) -> None:
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

    def test_asr_output_labels_show_voice_when_translation_disabled_and_tts_enabled(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "none"
        config.runtime.remote_tts_voice = "ja-JP-NanamiNeural"
        config.runtime.local_tts_voice = "th-TH-PremwadeeNeural"

        page.apply_config(config)
        page.update_translation_voice_labels(config)

        self.assertEqual(page.remote_translated_label.text(), "原音直通")
        self.assertEqual(page.local_translated_label.text(), "本地輸出")


if __name__ == "__main__":
    unittest.main()
