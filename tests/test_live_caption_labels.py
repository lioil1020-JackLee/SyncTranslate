from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.config.schema import AppConfig
from app.ui.pages.live_caption_page import LiveCaptionPage, _CHANNEL_DEFAULTS


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class LiveCaptionLabelTests(_QtTestCase):
    def test_original_panel_labels_follow_asr_language_family(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "zh-TW"
        config.runtime.local_asr_language = "en"
        config.asr_channels.local.model = r".\runtimes\models\belle-zh-ct2"
        config.asr_channels.remote.model = "large-v3-turbo"

        page.apply_config(config)

        self.assertIn("belle-zh-ct2", page.remote_original_label.text())
        self.assertIn("large-v3-turbo", page.local_original_label.text())

    def test_original_panel_model_label_updates_when_asr_language_changes(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "zh-TW"
        config.asr_channels.local.model = r".\runtimes\models\belle-zh-ct2"
        config.asr_channels.remote.model = "large-v3-turbo"

        page.apply_config(config)
        self.assertIn("belle-zh-ct2", page.remote_original_label.text())

        page.remote_asr_combo.setCurrentIndex(page.remote_asr_combo.findData("en"))

        self.assertIn("large-v3-turbo", page.remote_original_label.text())

    def test_legacy_auto_asr_config_is_normalized_to_fixed_ui_choice(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "auto"
        config.asr_channels.local.model = r".\runtimes\models\belle-zh-ct2"
        config.asr_channels.remote.model = "large-v3-turbo"

        page.apply_config(config)
        page.set_detected_asr_language("remote", "zh")

        self.assertEqual(page.remote_asr_combo.currentData(), "en")
        self.assertIn("large-v3-turbo", page.remote_original_label.text())

    def test_legacy_auto_asr_language_does_not_round_trip_through_product_ui(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "auto"

        page.apply_config(config)
        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.runtime.remote_asr_language, "en")

    def test_tts_mode_keeps_translated_panel_labels(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
        config.runtime.remote_translation_target = "ja"
        config.runtime.local_translation_target = "ko"
        config.runtime.remote_tts_voice = "ja-JP-NanamiNeural"
        config.runtime.local_tts_voice = "ko-KR-SunHiNeural"

        page.apply_config(config)
        page.update_translation_voice_labels(config)

        self.assertEqual(page.remote_translated_label.text(), _CHANNEL_DEFAULTS["remote"]["translated_label"])
        self.assertEqual(page.local_translated_label.text(), _CHANNEL_DEFAULTS["local"]["translated_label"])

    def test_voice_choice_alone_no_longer_switches_panel_into_translation_mode(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "none"
        config.runtime.remote_tts_voice = "ja-JP-NanamiNeural"
        config.runtime.local_tts_voice = "th-TH-PremwadeeNeural"

        page.apply_config(config)
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))
        page.local_tts_voice_combo.setCurrentIndex(page.local_tts_voice_combo.findData("none"))
        page.update_translation_voice_labels(config)

        self.assertEqual(page.remote_translated_label.text(), _CHANNEL_DEFAULTS["remote"]["output_label"])
        self.assertEqual(page.local_translated_label.text(), _CHANNEL_DEFAULTS["local"]["output_label"])

    def test_tts_none_round_trips_as_direct_passthrough(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
        config.runtime.remote_translation_target = "zh-TW"
        config.runtime.remote_tts_voice = "zh-TW-HsiaoChenNeural"
        config.runtime.remote_translation_enabled = True
        config.runtime.remote_tts_enabled = True
        config.dialogue.remote_to_local.output_policy = "translated_tts"

        page.apply_config(config)
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        updated = AppConfig()
        page.update_config(updated)

        reloaded = LiveCaptionPage()
        reloaded.apply_config(updated)

        self.assertEqual(updated.runtime.remote_translation_enabled, False)
        self.assertEqual(updated.runtime.remote_tts_enabled, False)
        self.assertEqual(updated.dialogue.remote_to_local.output_policy, "direct_passthrough")
        self.assertEqual(reloaded.selected_tts_output_mode_for_channel("remote"), "passthrough")


if __name__ == "__main__":
    unittest.main()
