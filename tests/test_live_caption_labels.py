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
    def test_auto_detected_asr_language_updates_original_panel_labels(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()

        page.apply_config(config)
        page.set_detected_asr_language("remote", "ja")
        page.set_detected_asr_language("local", "th")

        self.assertIn("自動偵測：日文", page.remote_original_label.text())
        self.assertIn("自動偵測：泰文", page.local_original_label.text())

    def test_tts_voice_labels_appear_on_translated_panels_when_tts_enabled(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.tts_output_mode = "tts"
        config.language.meeting_target = "ja"
        config.language.local_target = "ko"

        page.apply_config(config)
        page.update_translation_voice_labels(config)

        self.assertIn("Edge:", page.remote_translated_label.text())
        self.assertIn("ja-JP-NanamiNeural", page.remote_translated_label.text())
        self.assertIn("ko-KR-SunHiNeural", page.local_translated_label.text())

    def test_asr_output_labels_show_voice_when_translation_disabled_and_tts_enabled(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.tts_output_mode = "tts"
        config.runtime.remote_translation_enabled = False
        config.runtime.local_translation_enabled = False
        config.language.meeting_target = "ja"
        config.language.local_target = "th"

        page.apply_config(config)
        page.update_translation_voice_labels(config)

        self.assertIn("ASR 原文 / Edge:", page.remote_translated_label.text())
        self.assertIn("ja-JP-NanamiNeural", page.remote_translated_label.text())
        self.assertIn("th-TH-PremwadeeNeural", page.local_translated_label.text())


if __name__ == "__main__":
    unittest.main()
