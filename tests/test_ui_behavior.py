from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

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
        self.calls.append(("apply", str(getattr(audio, "meeting_in", "")), 1.0))


class LiveCaptionPageUiTests(_QtTestCase):
    def test_meeting_mode_hides_dialogue_controls_and_uses_meeting_labels(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "meeting"

        page.apply_config(config)

        self.assertEqual(page.remote_translated_label.text(), "會議翻譯")
        self.assertIn("會議原文", page.remote_original_label.text())
        self.assertFalse(page.remote_asr_combo.isHidden())
        self.assertFalse(page.remote_lang_combo.isHidden())
        self.assertTrue(page.remote_tts_voice_combo.isHidden())
        self.assertTrue(page.local_original.isHidden())
        self.assertTrue(page.local_tts_voice_combo.isHidden())
        self.assertLess(page.remote_asr_combo.findData("auto"), 0)

    def test_meeting_device_menu_preserves_windows_input_device_names(self) -> None:
        names = MainWindow._meeting_device_names(
            [
                SimpleNamespace(name="USB Microphone"),
                SimpleNamespace(name="Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"),
                SimpleNamespace(name="SyncTranslate Virtual Microphone"),
                SimpleNamespace(name="USB Microphone"),
            ]
        )

        self.assertEqual(
            names,
            [
                "USB Microphone",
                "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)",
                "SyncTranslate Virtual Microphone",
            ],
        )

    def test_session_mode_selector_lives_in_main_menu_bar(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        corner = window.tabs.cornerWidget(Qt.Corner.TopRightCorner)

        self.assertIsNotNone(corner)
        self.assertGreaterEqual(corner.layout().indexOf(window.live_caption_page.meeting_mode_btn), 0)
        self.assertGreaterEqual(corner.layout().indexOf(window.live_caption_page.dialogue_mode_btn), 0)
        self.assertTrue(window.live_caption_page.meeting_mode_btn.isCheckable())
        self.assertTrue(window.live_caption_page.dialogue_mode_btn.isCheckable())
        self.assertTrue(window.live_caption_page.session_mode_combo.isHidden())

    def test_mode_buttons_switch_session_pages(self) -> None:
        page = LiveCaptionPage()

        page.dialogue_mode_btn.click()
        self.assertEqual(page.selected_session_mode(), "dialogue")
        self.assertTrue(page.dialogue_mode_btn.isChecked())
        self.assertTrue(page.meeting_source_combo.isHidden())
        self.assertFalse(page.local_original.isHidden())

        page.meeting_mode_btn.click()
        self.assertEqual(page.selected_session_mode(), "meeting")
        self.assertTrue(page.meeting_mode_btn.isChecked())
        self.assertFalse(page.meeting_source_combo.isHidden())
        self.assertTrue(page.local_original.isHidden())

    def test_meeting_source_and_device_combos_fit_long_names(self) -> None:
        page = LiveCaptionPage()
        long_device = "Realtek USB Audio Microphone Array With Very Long Friendly Name"

        page.set_meeting_devices([long_device], [])

        metrics = page.meeting_device_combo.fontMetrics()
        self.assertGreaterEqual(
            page.meeting_device_combo.minimumWidth(),
            metrics.horizontalAdvance(long_device) + 40,
        )
        self.assertGreaterEqual(
            page.meeting_source_combo.minimumWidth(),
            page.meeting_source_combo.fontMetrics().horizontalAdvance("系統輸出 loopback") + 40,
        )

    def test_dialogue_notice_follows_tts_voice_rule(self) -> None:
        page = LiveCaptionPage()
        page.set_session_mode("dialogue")
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))

        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("zh-TW-HsiaoChenNeural"))
        self.assertIn("TTS 輸出", page.remote_policy_notice_label.text())

        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))
        self.assertIn("TTS語音=無", page.remote_policy_notice_label.text())
        self.assertIn("不辨識、不翻譯", page.remote_policy_notice_label.text())

    def test_main_window_asr_diagnostics_summary_surfaces_validator_rejections(self) -> None:
        router_stats = SimpleNamespace(
            asr={
                "remote": {
                    "queue_size": 0,
                    "queue_maxsize": 256,
                    "dropped_chunks": 0,
                    "partial_count": 4,
                    "final_count": 5,
                    "degradation_level": "normal",
                    "configured_model": "large-v3-turbo",
                    "endpoint_profile": "meeting_room",
                    "frontend_enhancement_enabled": True,
                    "endpoint_signal": {"pause_ms": 260},
                    "endpointing": {
                        "speech_started_count": 3,
                        "soft_endpoint_count": 4,
                        "hard_endpoint_count": 1,
                    },
                    "postprocessor": {
                        "final": {"rejected_count": 2, "last_rejection_reason": "markup-leak"},
                    },
                },
                "local": {
                    "queue_size": 1,
                    "queue_maxsize": 256,
                    "dropped_chunks": 2,
                    "partial_count": 2,
                    "final_count": 3,
                    "degradation_level": "congested",
                    "configured_model": r".\runtimes\models\belle-zh-ct2",
                    "endpoint_profile": "turn_taking",
                    "frontend_enhancement_enabled": False,
                    "endpoint_signal": {"pause_ms": 80},
                    "endpointing": {
                        "speech_started_count": 2,
                        "soft_endpoint_count": 1,
                        "hard_endpoint_count": 0,
                    },
                    "postprocessor": {
                        "final": {"rejected_count": 0, "last_rejection_reason": ""},
                    },
                },
            },
            capture={
                "local": {"running": True, "frame_count": 1200, "level": 0.015},
                "remote": {"running": True, "frame_count": 1195, "level": 0.014},
            },
            translation_overflow={"local": 0, "remote": 0},
            latency=[],
            tts={},
        )

        text = MainWindow._build_asr_diagnostics_summary(router_stats)

        self.assertIn("meeting:q=0/256 drop=0 p=4 f=5 pause=260 ep=3/4/1 deg=normal model=large-v3-turbo profile=meeting_room enh=on rej=2(markup-leak)", text)
        self.assertIn("local:q=1/256 drop=2 p=2 f=3 pause=80 ep=2/1/0 deg=congested model=belle-zh-ct2 profile=turn_taking enh=off rej=0", text)
        self.assertIn("capture=possible-same-source", text)

    def test_main_window_v2_runtime_summary_surfaces_productized_fields(self) -> None:
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
        config.meeting.audio_source = "system_output_loopback"
        config.dialogue.local_to_remote.output_policy = "direct_passthrough"
        config.dialogue.remote_to_local.output_policy = "translated_tts"
        router_stats = SimpleNamespace(
            state={"session_mode": "dialogue"},
            capture={"remote": {"sample_rate": 48000, "channels": 2}},
            bridge={"connected": True},
        )

        text = MainWindow._build_v2_runtime_summary(config, router_stats)

        self.assertIn("session_mode=dialogue", text)
        self.assertIn("meeting_audio_source=system_output_loopback", text)
        self.assertIn("capture_sample_rate=48000", text)
        self.assertIn("capture_channels=2", text)
        self.assertIn("asr_input_sample_rate=16000", text)
        self.assertIn("direct_passthrough_active_local=True", text)
        self.assertIn("bridge_required=True", text)
        self.assertIn("bridge_connected=True", text)

    def test_dialogue_readiness_warning_does_not_disable_start_button(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        window.live_caption_page.set_session_mode("dialogue")

        class _Readiness:
            summary = {
                "meeting_ready": True,
                "dialogue_ready": False,
                "asr_model_ready": True,
                "llm_model_ready": True,
                "bridge_ready": False,
                "driver_ready": False,
                "suggested_next_action": "Install driver",
            }

        import app.ui.main_window as main_window_module

        original = main_window_module.evaluate_first_run_readiness
        try:
            main_window_module.evaluate_first_run_readiness = lambda *args, **kwargs: _Readiness()
            window._refresh_first_run_readiness()
        finally:
            main_window_module.evaluate_first_run_readiness = original

        self.assertTrue(window._start_session_btn.isEnabled())

    def test_main_window_panel_status_uses_capture_running_as_started_state(self) -> None:
        captured: dict[str, str] = {}
        config = AppConfig()
        stats = SimpleNamespace(
            active_sources=["remote"],
            capture={
                "remote": {"running": True, "frame_count": 120},
                "local": {"running": False, "frame_count": 0},
            },
            asr={
                "remote": {"model_init_mode": "lazy", "partial_count": 0, "final_count": 0, "init_failure": ""},
                "local": {"model_init_mode": "lazy", "partial_count": 0, "final_count": 0, "init_failure": ""},
            },
        )

        class _Router:
            def stats(self):
                return stats

        class _Session:
            def is_running(self) -> bool:
                return True

        class _Page:
            def set_panel_statuses(self, **kwargs):
                captured.update(kwargs)

        class _Window:
            _resolve_active_sources = MainWindow._resolve_active_sources
            _resolve_source_runtime_state = MainWindow._resolve_source_runtime_state

            def __init__(self) -> None:
                self.audio_router = _Router()
                self.session_controller = _Session()
                self.live_caption_page = _Page()
                self.config = config
                self._session_action_running = False
                self._session_action_name = ""

        window = _Window()

        MainWindow._update_live_panel_statuses(
            window,  # type: ignore[arg-type]
            remote_original_active=False,
            remote_translated_active=False,
            local_original_active=False,
            local_translated_active=False,
        )

        self.assertEqual(captured["remote_original"], "running")
        self.assertEqual(captured["remote_translated"], "preparing")

    def test_main_window_translation_panel_runs_only_after_translation_output(self) -> None:
        captured: dict[str, str] = {}
        config = AppConfig()
        stats = SimpleNamespace(
            active_sources=["remote"],
            capture={
                "remote": {"running": True, "frame_count": 120},
                "local": {"running": False, "frame_count": 0},
            },
            asr={
                "remote": {"model_init_mode": "warm", "partial_count": 0, "final_count": 0, "init_failure": ""},
                "local": {"model_init_mode": "lazy", "partial_count": 0, "final_count": 0, "init_failure": ""},
            },
        )

        class _Router:
            def stats(self):
                return stats

        class _Session:
            def is_running(self) -> bool:
                return True

        class _Page:
            def set_panel_statuses(self, **kwargs):
                captured.update(kwargs)

        class _Window:
            _resolve_active_sources = MainWindow._resolve_active_sources
            _resolve_source_runtime_state = MainWindow._resolve_source_runtime_state

            def __init__(self) -> None:
                self.audio_router = _Router()
                self.session_controller = _Session()
                self.live_caption_page = _Page()
                self.config = config
                self._session_action_running = False
                self._session_action_name = ""

        window = _Window()

        MainWindow._update_live_panel_statuses(
            window,  # type: ignore[arg-type]
            remote_original_active=False,
            remote_translated_active=False,
            local_original_active=False,
            local_translated_active=False,
        )
        self.assertEqual(captured["remote_original"], "running")
        self.assertEqual(captured["remote_translated"], "preparing")

        MainWindow._update_live_panel_statuses(
            window,  # type: ignore[arg-type]
            remote_original_active=True,
            remote_translated_active=True,
            local_original_active=False,
            local_translated_active=False,
        )
        self.assertEqual(captured["remote_translated"], "running")

    def test_panel_status_maps_productized_router_source_names(self) -> None:
        captured: dict[str, str] = {}
        config = AppConfig()
        stats = SimpleNamespace(
            active_sources=["dialogue_remote", "dialogue_local"],
            capture={
                "remote": {"running": True, "frame_count": 120},
                "local": {"running": True, "frame_count": 140},
            },
            asr={
                "remote": {"model_init_mode": "warm", "partial_count": 0, "final_count": 0, "init_failure": ""},
                "local": {"model_init_mode": "warm", "partial_count": 0, "final_count": 0, "init_failure": ""},
            },
        )

        class _Router:
            def stats(self):
                return stats

        class _Session:
            def is_running(self) -> bool:
                return True

        class _Page:
            def set_panel_statuses(self, **kwargs):
                captured.update(kwargs)

        class _Window:
            _resolve_active_sources = MainWindow._resolve_active_sources
            _resolve_source_runtime_state = MainWindow._resolve_source_runtime_state

            def __init__(self) -> None:
                self.audio_router = _Router()
                self.session_controller = _Session()
                self.live_caption_page = _Page()
                self.config = config
                self._session_action_running = False
                self._session_action_name = ""

        MainWindow._update_live_panel_statuses(
            _Window(),  # type: ignore[arg-type]
            remote_original_active=False,
            remote_translated_active=False,
            local_original_active=False,
            local_translated_active=False,
        )

        self.assertEqual(captured["remote_original"], "running")
        self.assertEqual(captured["local_original"], "running")

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
        config.runtime.session_mode = "dialogue"
        config.dialogue.local_to_remote.output_policy = "translated_tts"
        config.language.meeting_target = "ja"
        config.language.local_target = "ko"

        page.apply_config(config)

        self.assertFalse(page.translation_enabled("remote"))
        self.assertTrue(page.translation_enabled("local"))
        self.assertEqual(page.selected_asr_language_mode(), "zh-TW")
        self.assertEqual(page.selected_tts_output_mode(), "tts")
        self.assertTrue(page.remote_lang_combo.isEnabled())
        self.assertTrue(page.local_lang_combo.isEnabled())
        self.assertIn("Translation target language", page.remote_lang_combo.toolTip())
        self.assertEqual(page.remote_translated_label.text(), "遠端輸出")
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")
        self.assertEqual(page.selected_tts_output_mode_for_channel("remote"), "passthrough")
        self.assertEqual(page.selected_tts_output_mode_for_channel("local"), "tts")
        self.assertEqual(page.selected_tts_output_mode(), "tts")
        self.assertFalse(hasattr(page, "asr_language_mode_combo"))

    def test_translation_labels_do_not_include_edge_in_status(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
        config.runtime.remote_translation_target = "zh-TW"
        config.runtime.local_translation_target = "en"
        config.runtime.remote_tts_voice = "zh-TW-HsiaoChenNeural"
        config.runtime.local_tts_voice = "en-US-JennyNeural"
        config.dialogue.remote_to_local.output_policy = "translated_tts"
        config.dialogue.local_to_remote.output_policy = "translated_tts"

        page.apply_config(config)

        self.assertEqual(page.remote_translated_label.text(), "遠端翻譯")
        self.assertEqual(page.local_translated_label.text(), "本地翻譯")

    def test_live_broadcast_mode_shown_when_no_translation_tts(self) -> None:
        page = LiveCaptionPage()
        config = AppConfig()
        config.runtime.session_mode = "dialogue"
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

    def test_tts_none_enters_direct_passthrough(self) -> None:
        page = LiveCaptionPage()
        page.session_mode_combo.setCurrentIndex(page.session_mode_combo.findData("dialogue"))
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        self.assertEqual(page.selected_tts_output_mode_for_channel("remote"), "passthrough")
        self.assertFalse(page.translation_enabled("remote"))

    def test_dialogue_controls_remain_editable_while_tts_none_direct_passthrough(self) -> None:
        page = LiveCaptionPage()
        page.set_session_mode("dialogue")
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        self.assertTrue(page.remote_asr_combo.isEnabled())
        self.assertTrue(page.remote_lang_combo.isEnabled())
        self.assertTrue(page.remote_tts_voice_combo.isEnabled())

    def test_asr_none_disables_translation_chain_for_channel(self) -> None:
        page = LiveCaptionPage()
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("none"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))

        self.assertFalse(page.translation_enabled("local"))
        self.assertEqual(page.selected_tts_output_mode_for_channel("local"), "subtitle_only")
        self.assertFalse(page.local_lang_combo.isEnabled())
        self.assertFalse(page.local_tts_voice_combo.isEnabled())

    def test_update_config_maps_tts_none_to_direct_passthrough(self) -> None:
        page = LiveCaptionPage()
        page.session_mode_combo.setCurrentIndex(page.session_mode_combo.findData("dialogue"))
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        updated = AppConfig()
        page.update_config(updated)

        self.assertFalse(updated.runtime.remote_translation_enabled)
        self.assertFalse(updated.runtime.remote_tts_enabled)
        self.assertEqual(updated.runtime.remote_translation_target, "zh-TW")
        self.assertEqual(updated.dialogue.remote_to_local.output_policy, "direct_passthrough")

    def test_meeting_tts_none_still_enables_translation_to_selected_target(self) -> None:
        page = LiveCaptionPage()
        page.set_session_mode("meeting")
        page.remote_asr_combo.setCurrentIndex(page.remote_asr_combo.findData("zh-TW"))
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("en"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("none"))

        updated = AppConfig()
        page.update_config(updated)

        self.assertTrue(updated.runtime.remote_translation_enabled)
        self.assertFalse(updated.runtime.remote_tts_enabled)
        self.assertEqual(updated.runtime.remote_translation_target, "en")
        self.assertEqual(updated.language.meeting_target, "en")
        self.assertEqual(updated.meeting.translation_target, "en")
        self.assertEqual(updated.dialogue.remote_to_local.output_policy, "subtitle_only")

    def test_update_config_with_asr_none_disables_channel_pipeline(self) -> None:
        page = LiveCaptionPage()
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("none"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.runtime.local_asr_language, "none")
        self.assertFalse(updated.runtime.local_translation_enabled)
        self.assertFalse(updated.runtime.local_tts_enabled)

    def test_selected_mode_reflects_single_direction_runtime(self) -> None:
        page = LiveCaptionPage()
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("none"))
        page.remote_asr_combo.setCurrentIndex(page.remote_asr_combo.findData("zh-TW"))

        self.assertEqual(page.selected_mode(), "meeting")

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.runtime.session_mode, "meeting")

    def test_selected_mode_remains_bidirectional_when_both_channels_are_active(self) -> None:
        page = LiveCaptionPage()
        page.session_mode_combo.setCurrentIndex(page.session_mode_combo.findData("dialogue"))
        page.local_asr_combo.setCurrentIndex(page.local_asr_combo.findData("zh-TW"))
        page.remote_asr_combo.setCurrentIndex(page.remote_asr_combo.findData("en"))

        self.assertEqual(page.selected_mode(), "dialogue")

    def test_direction_controls_lock_only_language_and_voice_controls(self) -> None:
        page = LiveCaptionPage()

        page.set_direction_controls_enabled(False)

        self.assertFalse(page.remote_asr_combo.isEnabled())
        self.assertFalse(page.local_tts_voice_combo.isEnabled())

    def test_direction_controls_can_stay_enabled_while_session_running(self) -> None:
        page = LiveCaptionPage()
        page.session_mode_combo.setCurrentIndex(page.session_mode_combo.findData("dialogue"))
        page.remote_lang_combo.setCurrentIndex(page.remote_lang_combo.findData("zh-TW"))
        page.remote_tts_voice_combo.setCurrentIndex(page.remote_tts_voice_combo.findData("zh-TW-HsiaoChenNeural"))
        page.local_lang_combo.setCurrentIndex(page.local_lang_combo.findData("en"))
        page.local_tts_voice_combo.setCurrentIndex(page.local_tts_voice_combo.findData("en-US-JennyNeural"))

        page.set_direction_controls_enabled(True)

        self.assertTrue(page.remote_asr_combo.isEnabled())
        self.assertTrue(page.remote_output_mode_combo.isHidden())
        self.assertTrue(page.local_tts_voice_combo.isEnabled())

    def test_four_panels_have_same_size(self) -> None:
        page = LiveCaptionPage()
        page.set_session_mode("dialogue")
        editors = (page.remote_original, page.remote_translated, page.local_original, page.local_translated)
        self.assertTrue(all(not editor.isHidden() for editor in editors))
        self.assertEqual({editor.minimumHeight() for editor in editors}, {160})

    def test_main_window_maps_remote_controls_to_local_output_and_local_controls_to_remote_output(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _Router:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def set_output_mode(self, channel: str, mode: str) -> None:
                self.calls.append((channel, mode))

        router = _Router()
        window.audio_router = router  # type: ignore[assignment]
        window.live_caption_page.session_mode_combo.setCurrentIndex(
            window.live_caption_page.session_mode_combo.findData("dialogue")
        )

        remote_asr_zh_index = window.live_caption_page.remote_asr_combo.findData("zh-TW")
        remote_zh_index = window.live_caption_page.remote_lang_combo.findData("zh-TW")
        local_en_index = window.live_caption_page.local_lang_combo.findData("en")
        local_asr_en_index = window.live_caption_page.local_asr_combo.findData("en")

        window.live_caption_page.remote_asr_combo.setCurrentIndex(remote_asr_zh_index)
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

    def test_main_window_has_no_audio_level_controls(self) -> None:
        controller = _DummyDeviceVolumeController()
        window = MainWindow("config.yaml", device_volume_controller=controller)
        self.assertFalse(hasattr(window.audio_routing_page, "meeting_in_gain_slider"))
        self.assertFalse(hasattr(window.audio_routing_page, "microphone_in_gain_slider"))
        self.assertFalse(hasattr(window.audio_routing_page, "speaker_out_volume_slider"))
        self.assertFalse(hasattr(window.audio_routing_page, "meeting_out_volume_slider"))
        self.assertEqual(controller.calls, [])

    def test_main_window_route_change_does_not_apply_in_app_volume(self) -> None:
        controller = _DummyDeviceVolumeController()
        window = MainWindow("config.yaml", device_volume_controller=controller)

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window.validate_current_routes()
        window._on_audio_routing_changed()

        self.assertEqual(controller.calls, [])

    def test_main_window_keeps_live_caption_controls_enabled_while_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        window.config.runtime.session_mode = "dialogue"
        window.live_caption_page.set_session_mode("dialogue")

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window.live_caption_page.remote_lang_combo.setCurrentIndex(
            window.live_caption_page.remote_lang_combo.findData("zh-TW")
        )
        window.live_caption_page.remote_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.remote_tts_voice_combo.findData("zh-TW-HsiaoChenNeural")
        )
        window.live_caption_page.local_asr_combo.setCurrentIndex(
            window.live_caption_page.local_asr_combo.findData("en")
        )
        window.live_caption_page.local_lang_combo.setCurrentIndex(
            window.live_caption_page.local_lang_combo.findData("en")
        )
        window.live_caption_page.local_tts_voice_combo.setCurrentIndex(
            window.live_caption_page.local_tts_voice_combo.findData("en-US-JennyNeural")
        )
        window.validate_current_routes()

        self.assertTrue(window.live_caption_page.remote_asr_combo.isEnabled())
        self.assertTrue(window.live_caption_page.remote_output_mode_combo.isHidden())
        self.assertTrue(window.live_caption_page.local_tts_voice_combo.isEnabled())


class AudioRoutingPageUiTests(_QtTestCase):
    def test_audio_route_page_no_longer_exposes_level_sliders(self) -> None:
        page = AudioRoutingPage(on_route_changed=lambda: None)
        config = AppConfig()

        page.apply_config(config)
        selected = page.selected_audio_routes()

        self.assertFalse(hasattr(page, "meeting_in_gain_slider"))
        self.assertFalse(hasattr(page, "microphone_in_gain_slider"))
        self.assertFalse(hasattr(page, "speaker_out_volume_slider"))
        self.assertFalse(hasattr(page, "meeting_out_volume_slider"))
        self.assertEqual(selected.routing_mode, "synctranslate_virtual_audio")

    def test_main_window_hot_applies_live_caption_settings_while_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        window.config.runtime.session_mode = "dialogue"
        window.live_caption_page.set_session_mode("dialogue")

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

        self.assertTrue(hasattr(page, "experience_preset_combo"))
        self.assertFalse(page.experience_preset_combo.isVisible())
        self.assertFalse(hasattr(page, "translation_style_combo"))
        self.assertFalse(hasattr(page, "tts_style_combo"))
        self.assertEqual(page.experience_preset_combo.currentData(), "meeting_monitor")
        self.assertEqual(page.experience_preset_combo.currentText(), "會議模式（穩定切段）")
        labels = [child.text() for child in page.experience_group.findChildren(QLabel)]
        self.assertNotIn("使用情境", labels)
        return
        self.assertIn("已內建高準確 + 低延遲", page.channel_strategy_label.text())

    def test_local_ai_page_uses_built_in_optimized_defaults(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        self.assertEqual(page.asr_model_combo.currentData(), "large-v3-turbo")
        self.assertEqual(page.remote_asr_model_combo.currentData(), "large-v3-turbo")
        self.assertTrue(page.asr_model_combo.isEnabled())
        self.assertTrue(page.remote_asr_model_combo.isEnabled())
        self.assertEqual(page.llm_model_label.text(), "hy-mt1.5-7b")
        self.assertEqual(page.remote_llm_model_label.text(), "hy-mt1.5-7b")
        self.assertEqual(page.asr_beam_spin.value(), 1)
        self.assertFalse(page.asr_condition_prev_check.isChecked())
        self.assertFalse(page.remote_asr_condition_prev_check.isChecked())
        self.assertEqual(page.asr_partial_interval_spin.value(), 520)
        self.assertEqual(page.remote_asr_partial_interval_spin.value(), 480)
        self.assertAlmostEqual(page.asr_rms_threshold_spin.value(), 0.022, places=3)
        self.assertAlmostEqual(page.remote_asr_rms_threshold_spin.value(), 0.020, places=3)
        self.assertEqual(page.asr_min_silence_spin.value(), 600)
        self.assertEqual(page.asr_speech_pad_spin.value(), 360)
        self.assertEqual(page.remote_asr_min_silence_spin.value(), 520)
        self.assertEqual(page.remote_asr_speech_pad_spin.value(), 320)
        self.assertEqual(page.runtime_sample_rate_spin.currentData(), 48000)
        self.assertEqual(page.runtime_chunk_spin.value(), 40)
        self.assertEqual(page.runtime_asr_pre_roll_spin.value(), 360)
        self.assertEqual(page.runtime_asr_partial_min_audio_spin.value(), 360)
        self.assertEqual(page.runtime_stable_partial_min_repeats_spin.value(), 3)
        self.assertEqual(page.runtime_partial_stability_delta_spin.value(), 6)
        self.assertFalse(page.runtime_early_final_check.isChecked())
        self.assertEqual(page.runtime_tts_max_wait_spin.value(), 2200)
        self.assertEqual(page.runtime_tts_cancel_policy_combo.currentData(), "older_only")
        self.assertEqual(page.asr_queue_local_spin.value(), 256)
        self.assertEqual(page.asr_queue_remote_spin.value(), 256)

        updated = AppConfig()
        page.update_config(updated)
        self.assertEqual(updated.asr_channels.local.engine, "faster_whisper")
        self.assertEqual(updated.asr_channels.remote.engine, "faster_whisper")
        self.assertEqual(updated.tts.style_preset, "broadcast_clear")
        self.assertEqual(updated.llm.caption_profile, "live_caption_fast")
        self.assertEqual(updated.llm.speech_profile, "speech_output_natural")
        self.assertEqual(updated.asr_channels.local.final_beam_size, 4)
        self.assertEqual(updated.asr_channels.remote.final_beam_size, 5)
        self.assertEqual(updated.runtime.asr_profile_local, "meeting_room")
        self.assertEqual(updated.runtime.asr_profile_remote, "meeting_room")
        self.assertIn("會議模式（穩定切段）", page._current_asr_runtime_hint_text())

    def test_local_ai_page_exports_selected_asr_model_routing(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._select_asr_model(page.asr_model_combo, r".\runtimes\models\belle-zh-ct2")

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.asr_channels.local.model, r".\runtimes\models\belle-zh-ct2")
        self.assertEqual(updated.asr_channels.remote.model, "large-v3-turbo")

    def test_local_ai_page_round_trips_advanced_runtime_and_profile_controls(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
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

    def test_local_ai_page_supports_dialogue_fast_profile(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        cfg = AppConfig()
        cfg.llm.caption_profile = "dialogue_fast"
        cfg.runtime.asr_profile_local = "turn_taking"
        cfg.runtime.asr_profile_remote = "turn_taking"

        page.apply_config(cfg)

        self.assertEqual(page.llm_caption_profile_combo.currentData(), "dialogue_fast")
        self.assertEqual(page.experience_preset_combo.currentData(), "dialogue")
        self.assertEqual(page.experience_preset_combo.currentText(), "對話模式（低延遲）")

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.llm.caption_profile, "dialogue_fast")
        self.assertEqual(updated.runtime.asr_profile_local, "turn_taking")
        self.assertEqual(updated.runtime.asr_profile_remote, "turn_taking")

    def test_local_ai_page_maps_dialogue_llm_profile_to_dialogue_preset(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        cfg = AppConfig()
        cfg.llm.caption_profile = "dialogue_fast"
        cfg.llm.speech_profile = "dialogue_fast"
        cfg.runtime.asr_profile_local = "meeting_room"
        cfg.runtime.asr_profile_remote = "meeting_room"

        page.apply_config(cfg)

        self.assertEqual(page.experience_preset_combo.currentData(), "dialogue")

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.llm.caption_profile, "dialogue_fast")
        self.assertEqual(updated.llm.speech_profile, "dialogue_fast")
        self.assertEqual(updated.runtime.asr_profile_local, "turn_taking")
        self.assertEqual(updated.runtime.asr_profile_remote, "turn_taking")

    def test_local_ai_page_keeps_final_correction_disabled_in_exported_runtime(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        updated = AppConfig()
        page.update_config(updated)

        self.assertFalse(updated.runtime.asr_final_correction_enabled)
        self.assertTrue(updated.runtime.asr_result_validator_enabled)
        self.assertTrue(updated.runtime.enable_postprocessor)

    def test_fixed_belle_model_gets_more_conservative_runtime_tuning(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)
        page._select_asr_model(page.asr_model_combo, r".\runtimes\models\belle-zh-ct2")

        updated = AppConfig()
        page.update_config(updated)

        self.assertEqual(updated.asr_channels.local.model, r".\runtimes\models\belle-zh-ct2")
        self.assertGreaterEqual(updated.asr_channels.local.beam_size, 2)
        self.assertGreaterEqual(updated.asr_channels.local.final_beam_size, 5)
        self.assertGreaterEqual(updated.asr_channels.local.streaming.final_history_seconds, 16)

    def test_switching_to_dialogue_preset_applies_conversation_values(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        page.experience_preset_combo.setCurrentIndex(page.experience_preset_combo.findData("dialogue"))

        self.assertEqual(page.asr_beam_spin.value(), 1)
        self.assertFalse(page.asr_condition_prev_check.isChecked())
        self.assertFalse(page.remote_asr_condition_prev_check.isChecked())
        self.assertEqual(page.asr_partial_interval_spin.value(), 240)
        self.assertEqual(page.remote_asr_partial_interval_spin.value(), 220)
        self.assertEqual(page.asr_min_silence_spin.value(), 280)
        self.assertEqual(page.remote_asr_min_silence_spin.value(), 260)
        self.assertEqual(page.llm_caption_profile_combo.currentData(), "dialogue_fast")
        self.assertEqual(page.runtime_latency_mode_combo.currentData(), "low_latency")
        self.assertTrue(page.runtime_early_final_check.isChecked())
        self.assertEqual(page.asr_queue_local_spin.value(), 256)
        self.assertEqual(page.asr_queue_remote_spin.value(), 256)
        self.assertIn("對話模式（低延遲）", page._current_asr_runtime_hint_text())

    def test_llm_model_is_fixed_for_both_directions(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        updated = AppConfig()
        page.update_config(updated)

        self.assertTrue(updated.runtime.use_channel_specific_llm)
        self.assertEqual(updated.llm.backend, "local_llama_inprocess")
        self.assertEqual(page.llm_backend_combo.currentData(), "local_llama_inprocess")
        self.assertIn("in-process", page.llm_backend_combo.currentText())
        self.assertEqual(page.llm_model_label.text(), "hy-mt1.5-7b")
        self.assertEqual(page.remote_llm_model_label.text(), "hy-mt1.5-7b")
        self.assertEqual(updated.llm_channels.local.model, "hy-mt1.5-7b")
        self.assertEqual(updated.llm_channels.remote.model, "hy-mt1.5-7b")

    def test_local_ai_page_exposes_quick_save_button(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        self.assertTrue(hasattr(page, "quick_save_btn"))
        self.assertIn(page.quick_save_btn, page.experience_group.findChildren(type(page.quick_save_btn)))

    def test_quick_save_and_advanced_toggle_still_exist(self) -> None:
        page = LocalAiPage(on_settings_changed=None, on_health_check=lambda: None, on_save_config=lambda: None)

        self.assertTrue(hasattr(page, "quick_save_btn"))
        self.assertTrue(hasattr(page, "show_advanced_check"))


class ASRObservationFragmentTests(_QtTestCase):
    """Tests for _build_asr_observation_fragment — effective params display (Section 5)."""

    def _asr(self, **overrides) -> dict:
        base: dict = {
            "queue_size": 0,
            "queue_maxsize": 256,
            "dropped_chunks": 0,
            "partial_count": 0,
            "final_count": 0,
            "degradation_level": "normal",
            "configured_model": "large-v3-turbo",
            "endpoint_profile": "meeting_room",
            "frontend_enhancement_enabled": False,
            "endpoint_signal": {"pause_ms": 0},
            "endpointing": {"speech_started_count": 0, "soft_endpoint_count": 0, "hard_endpoint_count": 0},
            "postprocessor": {"final": {"rejected_count": 0, "last_rejection_reason": ""}},
            "final_priority_active": False,
            "auto_language": {"requested": "", "effective": "", "detected": "", "family": "auto", "pending_rebuild": False},
        }
        base.update(overrides)
        return base

    def test_stats_includes_model_and_profile(self) -> None:
        asr = self._asr(configured_model="large-v3-turbo", endpoint_profile="meeting_room")
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertIn("model=large-v3-turbo", fragment)
        self.assertIn("profile=meeting_room", fragment)

    def test_stats_includes_queue_and_degradation(self) -> None:
        asr = self._asr(queue_size=12, queue_maxsize=256, degradation_level="congested")
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertIn("q=12/256", fragment)
        self.assertIn("deg=congested", fragment)

    def test_final_priority_active_shows_fp_on(self) -> None:
        asr = self._asr(final_priority_active=True)
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertIn("fp=on", fragment)

    def test_final_priority_inactive_no_fp_tag(self) -> None:
        asr = self._asr(final_priority_active=False)
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertNotIn("fp=", fragment)

    def test_legacy_auto_language_stats_are_not_primary_runtime_display(self) -> None:
        asr = self._asr(
            auto_language={"requested": "auto", "effective": "zh-TW", "detected": "zh", "family": "chinese", "pending_rebuild": False}
        )
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertNotIn("lang=auto", fragment)

    def test_legacy_auto_language_pending_rebuild_is_not_primary_runtime_display(self) -> None:
        asr = self._asr(
            auto_language={"requested": "auto", "effective": "zh-TW", "detected": "zh", "family": "chinese", "pending_rebuild": True}
        )
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertNotIn("lang=auto", fragment)
        self.assertNotIn("pending", fragment)

    def test_legacy_auto_language_no_effective_is_hidden(self) -> None:
        asr = self._asr(
            auto_language={"requested": "auto", "effective": "", "detected": "", "family": "auto", "pending_rebuild": False}
        )
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertNotIn("lang=auto", fragment)

    def test_non_auto_requested_language_hides_lang_tag(self) -> None:
        asr = self._asr(
            auto_language={"requested": "zh-TW", "effective": "zh-TW", "detected": "zh", "family": "chinese", "pending_rebuild": False}
        )
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertNotIn("lang=", fragment)

    def test_disabled_asr_empty_stats_does_not_crash(self) -> None:
        fragment = MainWindow._build_asr_observation_fragment({})
        self.assertIsInstance(fragment, str)

    def test_fp_display_ignores_legacy_auto_language_state(self) -> None:
        asr = self._asr(
            final_priority_active=True,
            auto_language={"requested": "auto", "effective": "zh-TW", "detected": "zh", "family": "chinese", "pending_rebuild": False},
        )
        fragment = MainWindow._build_asr_observation_fragment(asr)
        self.assertIn("fp=on", fragment)
        self.assertNotIn("lang=auto", fragment)



    """Tests that verify _schedule_live_apply and _apply_live_config_now behavior."""

    def test_schedule_live_apply_sets_pending_when_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window._pending_live_apply = False
        window._schedule_live_apply()

        self.assertTrue(window._pending_live_apply)
        # Timer must NOT be active (we deferred, not queued)
        self.assertFalse(window._live_apply_timer.isActive())

    def test_schedule_live_apply_starts_timer_when_session_idle(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _IdleSession:
            def is_running(self) -> bool:
                return False

        window.session_controller = _IdleSession()  # type: ignore[assignment]
        window._pending_live_apply = False
        window._schedule_live_apply()

        self.assertTrue(window._live_apply_timer.isActive())

    def test_apply_live_config_now_defers_when_session_running(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _RunningSession:
            def is_running(self) -> bool:
                return True

        window.session_controller = _RunningSession()  # type: ignore[assignment]
        window._pending_live_apply = False
        window._apply_live_config_now()

        self.assertTrue(window._pending_live_apply)

    def test_apply_live_config_now_applies_when_session_idle(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())

        class _IdleSession:
            def is_running(self) -> bool:
                return False

        window.session_controller = _IdleSession()  # type: ignore[assignment]
        window._pending_live_apply = True
        window._apply_live_config_now()

        # After applying, pending flag should be cleared
        self.assertFalse(window._pending_live_apply)

    def test_schedule_live_apply_is_no_op_when_not_ready(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        window._live_apply_ready = False
        window._pending_live_apply = False
        window._schedule_live_apply()

        self.assertFalse(window._pending_live_apply)
        self.assertFalse(window._live_apply_timer.isActive())

    def test_schedule_live_apply_is_no_op_when_suspended(self) -> None:
        window = MainWindow("config.yaml", device_volume_controller=_DummyDeviceVolumeController())
        window._suspend_live_apply = True
        window._pending_live_apply = False
        window._schedule_live_apply()

        self.assertFalse(window._pending_live_apply)
        self.assertFalse(window._live_apply_timer.isActive())


if __name__ == "__main__":
    unittest.main()
