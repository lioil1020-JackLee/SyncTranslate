from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.config.schema import AppConfig
from app.ui.main_window import MainWindow
from app.ui.pages.live_caption_page import LiveCaptionPage
from app.ui.pages.local_ai_page import LocalAiPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class LiveCaptionPageUiTests(_QtTestCase):
    def test_runtime_mode_controls_apply_to_labels_and_target_pickers(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_enabled = False
        config.runtime.local_translation_enabled = True
        config.runtime.tts_output_mode = "passthrough"
        config.language.meeting_target = "ja"
        config.language.local_target = "ko"

        page.apply_config(config)

        self.assertFalse(page.translation_enabled("remote"))
        self.assertTrue(page.translation_enabled("local"))
        self.assertEqual(page.selected_asr_language_mode(), "auto")
        self.assertEqual(page.selected_tts_output_mode(), "passthrough")
        self.assertTrue(page.remote_lang_combo.isEnabled())
        self.assertTrue(page.local_lang_combo.isEnabled())
        self.assertIn("來源語言由 ASR 自動偵測", page.remote_lang_combo.toolTip())
        self.assertIn("ASR 原文", page.remote_translated_label.text())
        self.assertIn("韓文", page.local_translated_label.text())
        self.assertFalse(hasattr(page, "asr_language_mode_combo"))


class LocalAiPageUiTests(_QtTestCase):
    def test_tts_style_combo_round_trips_with_config(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None)
        page._model_poll_timer.stop()
        page._reload_llm_models = lambda: None

        config = AppConfig()
        config.tts.style_preset = "fast_response"

        page.apply_config(config)
        self.assertEqual(page.tts_style_combo.currentData(), "fast_response")

        index = page.tts_style_combo.findData("broadcast_clear")
        self.assertGreaterEqual(index, 0)
        page.tts_style_combo.setCurrentIndex(index)

        updated = AppConfig()
        page.update_config(updated)
        self.assertEqual(updated.tts.style_preset, "broadcast_clear")
        self.assertEqual(updated.meeting_tts.style_preset, "broadcast_clear")
        self.assertEqual(updated.local_tts.style_preset, "broadcast_clear")

    def test_llm_models_loaded_from_lm_studio_can_be_selected_per_direction(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None)
        page._model_poll_timer.stop()
        page._model_loading = True
        page._model_load_queue.put((True, ["qwen-local", "qwen-remote"]))

        page._drain_model_load_queue()
        page.llm_model_combo.setCurrentText("qwen-local")
        page.remote_llm_model_combo.setCurrentText("qwen-remote")

        updated = AppConfig()
        page.update_config(updated)

        self.assertTrue(updated.runtime.use_channel_specific_llm)
        self.assertEqual(updated.llm_channels.local.model, "qwen-local")
        self.assertEqual(updated.llm_channels.remote.model, "qwen-remote")
        self.assertFalse(hasattr(page, "use_channel_specific_llm_check"))

    def test_local_output_tts_test_uses_remote_translation_target(self) -> None:
        config = AppConfig()
        config.language.meeting_target = "ja"
        config.language.local_target = "en"

        self.assertEqual(MainWindow._local_output_test_language(config), "ja")


if __name__ == "__main__":
    unittest.main()
