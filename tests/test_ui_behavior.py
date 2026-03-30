from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.infra.config.schema import AppConfig
from app.ui.main_window import MainWindow
from app.ui.pages.audio_routing_page import AudioRoutingPage
from app.ui.pages.live_caption_page import LiveCaptionPage
from app.ui.pages.local_ai_page import LocalAiPage


class _QtTestCase(unittest.TestCase):
    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])


class _DummyDeviceVolumeController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def set_input_volume(self, selector: str, scalar: float) -> None:
        self.calls.append(("input", selector, scalar))

    def set_output_volume(self, selector: str, scalar: float) -> None:
        self.calls.append(("output", selector, scalar))

    def apply_audio_route_config(self, audio) -> None:
        self.set_input_volume(audio.meeting_in, audio.meeting_in_gain)
        self.set_input_volume(audio.microphone_in, audio.microphone_in_gain)
        self.set_output_volume(audio.speaker_out, audio.speaker_out_volume)
        self.set_output_volume(audio.meeting_out, audio.meeting_out_volume)


class LiveCaptionPageUiTests(_QtTestCase):
    def test_runtime_mode_controls_apply_to_labels_and_target_pickers(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.remote_asr_language = "zh-TW"
        config.runtime.local_asr_language = "en"
        config.runtime.remote_translation_target = "none"
        config.runtime.local_translation_target = "ko"
        config.runtime.remote_tts_voice = "none"
        config.runtime.local_tts_voice = "ko-KR-SunHiNeural"
        config.runtime.local_tts_enabled = True
        config.language.meeting_target = "ja"
        config.language.local_target = "ko"

        page.apply_config(config)

        self.assertFalse(page.translation_enabled("remote"))
        self.assertTrue(page.translation_enabled("local"))
        self.assertEqual(page.selected_asr_language_mode(), "zh-TW")
        self.assertEqual(page.selected_tts_output_mode(), "tts")
        self.assertTrue(page.remote_lang_combo.isEnabled())
        self.assertTrue(page.local_lang_combo.isEnabled())
        self.assertIn("翻譯模式下使用的目標語言", page.remote_lang_combo.toolTip())
        self.assertEqual(page.remote_translated_label.text(), "遠端輸出")
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")
        self.assertEqual(page.selected_tts_output_mode_for_channel("remote"), "passthrough")
        self.assertEqual(page.selected_tts_output_mode_for_channel("local"), "tts")
        self.assertEqual(page.selected_tts_output_mode(), "tts")
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

    def test_passthrough_mode_does_not_disable_translation(self) -> None:
        page = LiveCaptionPage()
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        self.assertEqual(page.selected_tts_output_mode_for_channel("remote"), "passthrough")
        self.assertTrue(page.translation_enabled("remote"))

    def test_asr_none_disables_translation_chain_for_channel(self) -> None:
        page = LiveCaptionPage()
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("none"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))

        self.assertFalse(page.translation_enabled("local"))
        self.assertEqual(page.selected_tts_output_mode_for_channel("local"), "subtitle_only")
        self.assertFalse(page.local_lang_combo.isEnabled())
        self.assertFalse(page.local_tts_voice_combo.isEnabled())

    def test_update_config_preserves_translation_when_passthrough_is_selected(self) -> None:
        page = LiveCaptionPage()
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        updated = AppConfig()
        page.update_config(updated)

        self.assertTrue(updated.runtime.remote_translation_enabled)
        self.assertFalse(updated.runtime.remote_tts_enabled)
        self.assertEqual(updated.runtime.remote_translation_target, "zh-TW")
        self.assertEqual(updated.runtime.tts_output_mode, "passthrough")

    def test_update_config_with_asr_none_disables_channel_pipeline(self) -> None:
        page = LiveCaptionPage()
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("none"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.runtime.local_asr_language, "none")
        self.assertFalse(updated.runtime.local_translation_enabled)
        self.assertFalse(updated.runtime.local_tts_enabled)

    def test_direction_controls_lock_only_language_and_voice_controls(self) -> None:
        page = LiveCaptionPage()

        page.set_direction_controls_enabled(False)

        self.assertFalse(page.remote_asr_combo.isEnabled())
        self.assertFalse(page.local_tts_voice_combo.isEnabled())

    def test_direction_controls_can_stay_enabled_while_session_running(self) -> None:
        page = LiveCaptionPage()
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))

        page.set_direction_controls_enabled(True)

        self.assertTrue(page.remote_asr_combo.isEnabled())
        self.assertFalse(page.remote_output_mode_combo.isVisible())
        self.assertTrue(page.local_tts_voice_combo.isEnabled())

    def test_four_panels_have_same_size(self) -> None:
        page = LiveCaptionPage()
        self.assertEqual(page.remote_original.size(), page.remote_translated.size())
        self.assertEqual(page.remote_original.size(), page.local_original.size())
        self.assertEqual(page.remote_original.size(), page.local_translated.size())

    def test_main_window_maps_remote_controls_to_local_output_and_local_controls_to_remote_output(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _Router:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def set_output_mode(self, channel: str, mode: str) -> None:
                self.calls.append((channel, mode))

        router = _Router()
        window.audio_router = router  # type: ignore[assignment]

        remote_zh_index = window.live_caption_page.remote_lang_combo.findData("zh-TW")
        local_en_index = window.live_caption_page.local_lang_combo.findData("en")
        local_asr_en_index = window.live_caption_page.local_asr_combo.findData("en")

        window.live_caption_page.remote_lang_combo.setCurrentIndex(remote_zh_index)
        window.live_caption_page.local_asr_combo.setCurrentIndex(local_asr_en_index)
        window.live_caption_page.local_lang_combo.setCurrentIndex(local_en_index)
        window.live_caption_page._on_translation_target_changed()
        window.live_caption_page.remote_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.remote_tts_voice_combo.findData("zh-TW-HsiaoChenNeural")
        )
        window.live_caption_page.local_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.local_tts_voice_combo.findData("none")
        )

        window._apply_output_switches_to_router()

        self.assertIn(("local", "tts"), router.calls)
        self.assertIn(("remote", "passthrough"), router.calls)

    def test_main_window_applies_audio_route_levels_to_inputs_and_outputs(self) -> None:
        controller = _DummyDeviceVolumeController()
        window = MainWindow("config.yaml", device_volume_controller=controller)
        window.audio_routing_page.meeting_in_gain_slider.setValue(30)
        window.audio_routing_page.microphone_in_gain_slider.setValue(40)
        window.audio_routing_page.speaker_out_volume_slider.setValue(50)
        window.audio_routing_page.meeting_out_volume_slider.setValue(60)

        window._sync_ui_to_config()

        self.assertAlmostEqual(window.config.audio.meeting_in_gain, 0.3, places=2)
        self.assertAlmostEqual(window.config.audio.microphone_in_gain, 0.4, places=2)
        self.assertAlmostEqual(window.config.audio.speaker_out_volume, 0.5, places=2)
        self.assertAlmostEqual(window.config.audio.meeting_out_volume, 0.6, places=2)
        self.assertAlmostEqual(window.meeting_capture._gain, 1.0, places=2)
        self.assertAlmostEqual(window.local_capture._gain, 1.0, places=2)
        self.assertAlmostEqual(window.speaker_playback._volume, 1.0, places=2)
        self.assertAlmostEqual(window.meeting_playback._volume, 1.0, places=2)
        self.assertTrue(any(call[0] == "output" and abs(call[2] - 0.5) < 1e-6 for call in controller.calls))

    def test_main_window_applies_audio_route_changes_even_while_session_running(self) -> None:
        controller = _DummyDeviceVolumeController()
        window = MainWindow("config.yaml", device_volume_controller=controller)

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window.validate_current_routes()
        window.audio_routing_page.speaker_out_volume_slider.setValue(80)
        window.audio_routing_page.meeting_out_volume_slider.setValue(90)

        self.assertAlmostEqual(window.config.audio.speaker_out_volume, 0.8, places=2)
        self.assertAlmostEqual(window.config.audio.meeting_out_volume, 0.9, places=2)
        self.assertAlmostEqual(window.speaker_playback._volume, 1.0, places=2)
        self.assertAlmostEqual(window.meeting_playback._volume, 1.0, places=2)
        self.assertTrue(any(call[0] == "output" and abs(call[2] - 0.9) < 1e-6 for call in controller.calls))

    def test_main_window_keeps_live_caption_controls_enabled_while_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window.live_caption_page.remote_lang_combo.setCurrentIndex(
            window.live_caption_page.remote_lang_combo.findData("zh-TW")
        )
        window.live_caption_page.local_asr_combo.setCurrentIndex(
            window.live_caption_page.local_asr_combo.findData("en")
        )
        window.live_caption_page.local_lang_combo.setCurrentIndex(
            window.live_caption_page.local_lang_combo.findData("en")
        )
        window.validate_current_routes()

        self.assertTrue(window.live_caption_page.remote_asr_combo.isEnabled())
        self.assertFalse(window.live_caption_page.remote_output_mode_combo.isVisible())
        self.assertTrue(window.live_caption_page.local_tts_voice_combo.isEnabled())


class AudioRoutingPageUiTests(_QtTestCase):
    def test_audio_route_sliders_round_trip_with_config(self) -> None:
        page = AudioRoutingPage(on_route_changed=lambda: None)
        config = AppConfig()
        config.audio.meeting_in_gain = 0.3
        config.audio.microphone_in_gain = 0.4
        config.audio.speaker_out_volume = 0.5
        config.audio.meeting_out_volume = 0.6

        page.apply_config(config)
        selected = page.selected_audio_routes()

        self.assertAlmostEqual(selected.meeting_in_gain, 0.3, places=2)
        self.assertAlmostEqual(selected.microphone_in_gain, 0.4, places=2)
        self.assertAlmostEqual(selected.speaker_out_volume, 0.5, places=2)
        self.assertAlmostEqual(selected.meeting_out_volume, 0.6, places=2)
        self.assertEqual(page.meeting_out_volume_value_label.text(), "60%")

    def test_main_window_hot_applies_live_caption_settings_while_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        class _Router:
            def __init__(self) -> None:
                self.set_output_calls: list[tuple[str, str]] = []
                self.refresh_calls: list[object] = []

            def set_output_mode(self, channel: str, mode: str) -> None:
                self.set_output_calls.append((channel, mode))

            def refresh_runtime_config(self, config) -> None:
                self.refresh_calls.append(config)

        router = _Router()
        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window.audio_router = router  # type: ignore[assignment]
        window.validate_current_routes()

        window.live_caption_page.remote_lang_combo.setCurrentIndex(
            window.live_caption_page.remote_lang_combo.findData("en")
        )
        window.live_caption_page.remote_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.remote_tts_voice_combo.findData("en-US-GuyNeural")
        )
        window.live_caption_page.remote_asr_combo.setCurrentIndex(
            window.live_caption_page.remote_asr_combo.findData("ja")
        )

        window._apply_live_caption_config_now()

        self.assertEqual(window.config.runtime.remote_translation_target, "en")
        self.assertEqual(window.config.language.meeting_target, "en")
        self.assertEqual(window.config.runtime.remote_tts_voice, "en-US-GuyNeural")
        self.assertEqual(window.config.runtime.remote_asr_language, "ja")
        self.assertIn(("local", "tts"), router.set_output_calls)
        self.assertIs(router.refresh_calls[-1], window.config)


class LocalAiPageUiTests(_QtTestCase):
    def test_local_ai_page_hides_quick_tuning_combos(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()

        self.assertFalse(hasattr(page, "experience_preset_combo"))
        self.assertFalse(hasattr(page, "translation_style_combo"))
        self.assertFalse(hasattr(page, "tts_style_combo"))
        self.assertIn("已內建高準確 + 低延遲", page.channel_strategy_label.text())

    def test_local_ai_page_uses_built_in_optimized_defaults(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()

        self.assertEqual(page.asr_model_combo.currentText(), "large-v3")
        self.assertEqual(page.remote_asr_model_combo.currentText(), "large-v3")
        self.assertEqual(page.asr_beam_spin.value(), 3)
        self.assertFalse(page.asr_condition_prev_check.isChecked())
        self.assertFalse(page.remote_asr_condition_prev_check.isChecked())
        self.assertEqual(page.asr_partial_interval_spin.value(), 800)
        self.assertEqual(page.remote_asr_partial_interval_spin.value(), 800)
        self.assertAlmostEqual(page.asr_rms_threshold_spin.value(), 0.02, places=3)
        self.assertAlmostEqual(page.remote_asr_rms_threshold_spin.value(), 0.02, places=3)
        self.assertEqual(page.runtime_sample_rate_spin.currentData(), 48000)
        self.assertEqual(page.runtime_chunk_spin.value(), 40)
        self.assertEqual(page.runtime_asr_partial_min_audio_spin.value(), 520)
        self.assertEqual(page.runtime_tts_max_wait_spin.value(), 2200)
        self.assertEqual(page.runtime_tts_cancel_policy_combo.currentData(), "older_only")

        updated = AppConfig()
        page.update_config(updated)
        self.assertEqual(updated.tts.style_preset, "broadcast_clear")
        self.assertEqual(updated.llm.caption_profile, "live_caption_fast")
        self.assertEqual(updated.llm.speech_profile, "speech_output_natural")

    def test_local_ai_page_round_trips_advanced_runtime_and_profile_controls(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()
        cfg = AppConfig()
        cfg.runtime.latency_mode = "low_latency"
        cfg.runtime.display_partial_strategy = "all"
        cfg.runtime.stable_partial_min_repeats = 3
        cfg.runtime.partial_stability_max_delta_chars = 11
        cfg.runtime.asr_partial_min_audio_ms = 310
        cfg.runtime.asr_partial_interval_floor_ms = 260
        cfg.runtime.llm_partial_interval_floor_ms = 290
        cfg.runtime.early_final_enabled = False
        cfg.runtime.tts_accept_stable_partial = False
        cfg.runtime.tts_partial_min_chars = 18
        cfg.runtime.tts_use_speech_profile = True
        cfg.runtime.local_echo_guard_enabled = True
        cfg.runtime.local_echo_guard_resume_delay_ms = 450
        cfg.runtime.remote_echo_guard_resume_delay_ms = 550
        cfg.llm.caption_profile = "technical_meeting"
        cfg.llm.speech_profile = "speech_output_natural"
        cfg.tts.style_preset = "conversational"

        page.apply_config(cfg)

        self.assertEqual(page.runtime_latency_mode_combo.currentData(), "low_latency")
        self.assertEqual(page.runtime_display_partial_strategy_combo.currentData(), "all")
        self.assertEqual(page.llm_caption_profile_combo.currentData(), "technical_meeting")
        self.assertEqual(page.base_tts.style_preset_combo.currentData(), "conversational")
        self.assertTrue(page.runtime_tts_use_speech_profile_check.isChecked())

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.runtime.latency_mode, "low_latency")
        self.assertEqual(updated.runtime.display_partial_strategy, "all")
        self.assertEqual(updated.runtime.stable_partial_min_repeats, 3)
        self.assertEqual(updated.runtime.partial_stability_max_delta_chars, 11)
        self.assertEqual(updated.runtime.asr_partial_min_audio_ms, 310)
        self.assertEqual(updated.runtime.asr_partial_interval_floor_ms, 260)
        self.assertEqual(updated.runtime.llm_partial_interval_floor_ms, 290)
        self.assertFalse(updated.runtime.early_final_enabled)
        self.assertFalse(updated.runtime.tts_accept_stable_partial)
        self.assertEqual(updated.runtime.tts_partial_min_chars, 18)
        self.assertTrue(updated.runtime.tts_use_speech_profile)
        self.assertTrue(updated.runtime.local_echo_guard_enabled)
        self.assertEqual(updated.runtime.local_echo_guard_resume_delay_ms, 450)
        self.assertEqual(updated.runtime.remote_echo_guard_resume_delay_ms, 550)
        self.assertEqual(updated.llm.caption_profile, "technical_meeting")
        self.assertEqual(updated.llm.speech_profile, "speech_output_natural")
        self.assertEqual(updated.tts.style_preset, "conversational")

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

    def test_local_ai_page_exposes_quick_save_button(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()

        self.assertTrue(hasattr(page, "quick_save_btn"))
        self.assertIn(page.quick_save_btn, page.experience_group.findChildren(type(page.quick_save_btn)))

    def test_quick_save_and_advanced_toggle_still_exist(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._model_poll_timer.stop()

        self.assertTrue(hasattr(page, "quick_save_btn"))
        self.assertTrue(hasattr(page, "show_advanced_check"))


if __name__ == "__main__":
    unittest.main()
