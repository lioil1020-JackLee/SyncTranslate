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
        config.runtime.remote_asr_language = "zh-TW"
        config.runtime.local_asr_language = "en"
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "ko"
        config.runtime.remote_tts_voice = "none"
        config.runtime.local_tts_voice = "none"
        config.language.meeting_target = "ja"
        config.language.local_target = "ko"

        page.apply_config(config)

        self.assertFalse(page.translation_enabled("remote"))
        self.assertTrue(page.translation_enabled("local"))
        self.assertEqual(page.selected_asr_language_mode(), "zh-TW")
        self.assertEqual(page.selected_tts_output_mode(), "subtitle_only")
        self.assertTrue(page.remote_lang_combo.isEnabled())
        self.assertTrue(page.local_lang_combo.isEnabled())
        self.assertIn("LLM翻譯目標", page.remote_lang_combo.toolTip())
        self.assertIn("遠端輸出", page.remote_translated_label.text())
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")
        self.assertEqual(page.selected_tts_output_mode_for_channel("remote"), "passthrough")
        self.assertEqual(page.selected_tts_output_mode_for_channel("local"), "subtitle_only")
        self.assertEqual(page.selected_tts_output_mode(), "subtitle_only")
        self.assertFalse(hasattr(page, "asr_language_mode_combo"))

    def test_translation_labels_do_not_include_edge_in_status(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "zh-TW"
        config.runtime.local_translation_target = "en"
        config.runtime.remote_tts_voice = "zh-TW-HsiaoChenNeural"
        config.runtime.local_tts_voice = "en-US-JennyNeural"

        page.apply_config(config)

        self.assertEqual(page.remote_translated_label.text(), "遠端翻譯")
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")

    def test_live_broadcast_mode_shown_when_no_translation_tts(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "none"
        config.runtime.remote_tts_voice = "none"
        config.runtime.local_tts_voice = "none"

        page.apply_config(config)

        self.assertEqual(page.remote_translated_label.text(), "遠端輸出")
        self.assertEqual(page.local_translated_label.text(), "本地輸出")

    def test_tts_voice_combo_populates_based_on_translation_target(self) -> None:
        page = LiveCaptionPage()
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))
        page._on_translation_target_changed()

        self.assertGreater(page.remote_tts_voice_combo.count(), 1)
        self.assertEqual(page.remote_tts_voice_combo.itemData(0), "none")
        self.assertNotEqual(page.remote_tts_voice_combo.itemData(0), page.remote_tts_voice_combo.itemData(1))

    def test_four_panels_have_same_size(self) -> None:
        page = LiveCaptionPage()
        self.assertEqual(page.remote_original.size(), page.remote_translated.size())
        self.assertEqual(page.remote_original.size(), page.local_original.size())
        self.assertEqual(page.remote_original.size(), page.local_translated.size())

    def test_main_window_maps_remote_controls_to_local_output_and_local_controls_to_remote_output(self) -> None:
        window = MainWindow("config.yaml")

        class _Router:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def set_output_mode(self, channel: str, mode: str) -> None:
                self.calls.append((channel, mode))

        router = _Router()
        window.audio_router = router  # type: ignore[assignment]

        remote_none_index = window.live_caption_page.remote_lang_combo.findData("none")
        local_none_index = window.live_caption_page.local_lang_combo.findData("none")
        remote_zh_index = window.live_caption_page.remote_lang_combo.findData("zh-TW")
        local_en_index = window.live_caption_page.local_lang_combo.findData("en")

        window.live_caption_page.remote_lang_combo.setCurrentIndex(remote_zh_index)
        window.live_caption_page.local_lang_combo.setCurrentIndex(local_none_index)
        window.live_caption_page._on_translation_target_changed()
        window.live_caption_page.remote_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.remote_tts_voice_combo.findData("none")
        )
        window.live_caption_page.local_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.local_tts_voice_combo.findData("none")
        )

        window._apply_output_switches_to_router()

        self.assertIn(("local", "subtitle_only"), router.calls)
        self.assertIn(("remote", "passthrough"), router.calls)


class LocalAiPageUiTests(_QtTestCase):
    def test_tts_style_combo_round_trips_with_config(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
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
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
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

    def test_local_output_tts_uses_selected_remote_voice(self) -> None:
        window = MainWindow("config.yaml")
        window.config.language.meeting_target = "en"
        window.live_caption_page.apply_config(window.config)
        remote_target_index = window.live_caption_page.remote_lang_combo.findData("en")
        window.live_caption_page.remote_lang_combo.setCurrentIndex(remote_target_index)
        window.live_caption_page._on_translation_target_changed()
        remote_voice_index = window.live_caption_page.remote_tts_voice_combo.findData("en-US-GuyNeural")
        window.live_caption_page.remote_tts_voice_combo.setCurrentIndex(remote_voice_index)

        called = {}
        def fake_build_output_test_audio(*, primary_tts, text):
            called['voice'] = primary_tts.voice_name
            return (b"", 22050, "edge_tts")

        window._build_output_test_audio = fake_build_output_test_audio
        window.speaker_playback.play = lambda audio, sample_rate, output_device_name, blocking: None

        window.test_local_tts_output()
        self.assertEqual(called.get('voice'), "en-US-GuyNeural")

    def test_local_ai_page_exposes_quick_save_button(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()

        self.assertTrue(hasattr(page, "quick_save_btn"))
        self.assertIn(page.quick_save_btn, page.experience_group.findChildren(type(page.quick_save_btn)))

    def test_experience_fast_preset_updates_runtime_and_queue_defaults(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()
        index = page.experience_preset_combo.findData("fast")
        self.assertGreaterEqual(index, 0)

        page.experience_preset_combo.setCurrentIndex(index)

        self.assertEqual(page.asr_model_combo.currentText(), "large-v3-turbo")
        self.assertEqual(page.remote_asr_model_combo.currentText(), "large-v3-turbo")
        self.assertEqual(page.asr_queue_local_spin.value(), 64)
        self.assertEqual(page.llm_queue_local_spin.value(), 16)
        self.assertEqual(page.runtime_chunk_spin.value(), 30)
        self.assertEqual(page.runtime_tts_max_wait_spin.value(), 1500)

    def test_tts_style_updates_runtime_tts_controls(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()
        index = page.tts_style_combo.findData("fast_response")
        self.assertGreaterEqual(index, 0)

        page.tts_style_combo.setCurrentIndex(index)

        self.assertEqual(page.runtime_tts_max_wait_spin.value(), 1400)
        self.assertEqual(page.runtime_tts_max_chars_spin.value(), 120)
        self.assertEqual(page.runtime_tts_drop_threshold_spin.value(), 3)
        self.assertEqual(page.runtime_tts_cancel_policy_combo.currentData(), "older_only")


if __name__ == "__main__":
    unittest.main()
