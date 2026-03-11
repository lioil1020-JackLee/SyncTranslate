from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from threading import Thread
import time
from typing import Callable

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication, QShowEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.model_factory import (
    OPENAI_COMPATIBLE_PROVIDERS,
    create_asr_provider,
    create_translate_provider,
    create_tts_provider,
)
from app.pipeline_local import LocalPipeline
from app.pipeline_remote import RemotePipeline
from app.debug_panel import DebugPanel
from app.device_manager import DeviceManager
from app.env_vars import get_env_var
from app.pages_audio_routing import AudioRoutingPage
from app.pages_diagnostics import DiagnosticsPage
from app.pages_live_caption import LiveCaptionPage
from app.pages_models import ModelsPage
from app.pages_quick_start import QuickStartPage
from app.router import check_routes
from app.schemas import AppConfig
from app.session_controller import SessionController
from app.settings import load_config, save_config
from app.transcript_buffer import TranscriptBuffer


class MainWindow(QMainWindow):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = config_path
        self.config: AppConfig = load_config(self.config_path)
        self.device_manager = DeviceManager()
        self.remote_capture = AudioCapture()
        self.local_capture = AudioCapture()
        self.audio_playback = AudioPlayback()
        self.meeting_audio_playback = AudioPlayback()
        self.transcript_buffer = TranscriptBuffer(max_items=300)
        self.recent_errors: list[str] = []
        self._error_lock = Lock()
        self.remote_pipeline: RemotePipeline | None = None
        self.local_pipeline: LocalPipeline | None = None
        self.session_controller: SessionController | None = None
        self._provider_test_running = False
        self._provider_test_queue: Queue[tuple[int, str, bool, str]] = Queue()
        self._provider_test_token = 0
        self._provider_test_active_token: int | None = None
        self._provider_test_started_at = 0.0
        self._provider_test_timeout_sec = 25.0
        self._last_success_preview_chars = 72
        self._provider_test_last_success: dict[str, str] = {}
        self._load_provider_test_state_from_config()
        self.test_tts_provider = create_tts_provider(
            provider_name=self.config.model.tts_provider,
            openai_api_key_env=self.config.openai.api_key_env,
            openai_base_url=self.config.openai.base_url,
            openai_model=self.config.openai.tts_model,
            openai_voice=self.config.openai.tts_voice,
        )

        self.setWindowTitle("SyncTranslate - Stage 6 UI Skeleton")
        self._set_initial_window_geometry()

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.quick_start_page = QuickStartPage(self.apply_banana_preset)
        self.audio_routing_page = AudioRoutingPage(
            on_apply_banana_preset=self.apply_banana_preset,
            on_save=self.persist_config,
            on_reload=self.reload_config,
            on_route_changed=self.validate_current_routes,
            on_start=self.start_session,
        )
        self.live_caption_page = LiveCaptionPage()
        self.models_page = ModelsPage(
            on_test_asr=self.test_asr_provider,
            on_test_translate=self.test_translate_provider,
            on_test_tts=self.test_tts_provider_call,
            on_cancel_test=self.cancel_provider_test,
            on_clear_test_state=self.clear_provider_test_state,
            on_settings_changed=self._on_models_settings_changed,
        )
        self.diagnostics_page = DiagnosticsPage(
            on_start_remote_capture=self.start_remote_capture,
            on_stop_remote_capture=self.stop_remote_capture,
            on_start_local_capture=self.start_local_capture,
            on_stop_local_capture=self.stop_local_capture,
            on_set_local_mute=self.set_local_mute,
            on_rebind_local_capture=self.rebind_local_capture,
            on_test_meeting_tts=self.test_meeting_tts_output,
            on_export_diagnostics=self.export_diagnostics,
            remote_stats_provider=self.remote_capture.stats,
            local_stats_provider=self.local_capture.stats,
        )
        self.debug_panel = DebugPanel()

        self.tabs.addTab(self._wrap_in_scroll_area(self.quick_start_page), "快速開始")
        self.tabs.addTab(self._wrap_in_scroll_area(self.audio_routing_page), "音訊路由")
        self.tabs.addTab(self._wrap_in_scroll_area(self.live_caption_page), "即時字幕")
        self.tabs.addTab(self._wrap_in_scroll_area(self.models_page), "模型與語言")
        self.tabs.addTab(self._wrap_in_scroll_area(self.diagnostics_page), "音訊診斷")
        self.tabs.addTab(self._wrap_in_scroll_area(self.debug_panel), "除錯與診斷")
        self.setCentralWidget(self.tabs)
        self._build_pipelines_from_config()

        self.input_device_names: set[str] = set()
        self.output_device_names: set[str] = set()
        self._remote_line_cache: list[str] = []
        self._remote_translated_line_cache: list[str] = []
        self._local_line_cache: list[str] = []
        self._local_translated_line_cache: list[str] = []

        self.caption_timer = QTimer(self)
        self.caption_timer.setInterval(300)
        self.caption_timer.timeout.connect(self.refresh_live_caption)
        self.caption_timer.start()
        self.refresh_from_system()

    def refresh_from_system(self) -> None:
        input_devices = self.device_manager.list_input_devices()
        output_devices = self.device_manager.list_output_devices()
        voicemeeter_devices = self.device_manager.find_voicemeeter_devices()
        self.input_device_names = {d.name for d in input_devices}
        self.output_device_names = {d.name for d in output_devices}

        self.audio_routing_page.set_devices(input_devices, output_devices)
        self.audio_routing_page.apply_config(self.config)
        self.models_page.apply_config(self.config)
        self._update_model_runtime_status()
        self.diagnostics_page.set_input_devices(input_devices)
        self.quick_start_page.set_detected_voicemeeter_devices([d.name for d in voicemeeter_devices])
        self.statusBar().showMessage(
            f"Config: {self.config_path} | input={len(input_devices)} output={len(output_devices)} "
            f"voicemeeter={len(voicemeeter_devices)}"
        )
        self.validate_current_routes()

    def apply_banana_preset(self) -> None:
        self.config.audio.remote_in = "VoiceMeeter Aux Output (VB-Audio VoiceMeeter AUX VAIO)"
        self.config.audio.meeting_tts_out = "VoiceMeeter AUX Input (VB-Audio VoiceMeeter AUX VAIO)"
        self.audio_routing_page.apply_config(self.config)
        self.audio_routing_page.set_status("已套用 Banana 預設（請確認 local_mic_in / local_tts_out）")
        self.validate_current_routes()

    def persist_config(self) -> None:
        self._sync_ui_to_config()
        self._build_pipelines_from_config()
        self._update_model_runtime_status()
        path = self._save_config_to_disk()
        self.audio_routing_page.set_status(f"設定已儲存: {path}")
        self.statusBar().showMessage(f"已儲存設定到 {path}")
        self.validate_current_routes()

    def reload_config(self) -> None:
        if self.session_controller:
            self.session_controller.stop()
        self.config = load_config(self.config_path)
        self._load_provider_test_state_from_config()
        self._build_pipelines_from_config()
        self._update_model_runtime_status()
        self.refresh_from_system()
        self.audio_routing_page.set_status(f"已重載設定: {self.config_path}")

    def validate_current_routes(self) -> None:
        current_routes = self.audio_routing_page.selected_audio_routes()
        current_mode = self.audio_routing_page.selected_mode()
        self.diagnostics_page.select_remote_in(current_routes.remote_in)
        self.diagnostics_page.select_local_mic(current_routes.local_mic_in)
        result = check_routes(
            current_routes,
            self.input_device_names,
            self.output_device_names,
            mode=current_mode,
        )
        self.debug_panel.update_route_result(result)
        self.debug_panel.update_recent_errors(self._get_recent_errors())
        is_running = self.session_controller.is_running() if self.session_controller else False
        self.audio_routing_page.set_start_enabled(result.ok or is_running)
        self.audio_routing_page.set_start_label("停止" if is_running else "開始")
        if result.ok:
            self.audio_routing_page.set_status("路由檢查: OK")
        else:
            self.audio_routing_page.set_status("路由檢查: Error，請到除錯與診斷頁查看")

    def start_session(self) -> None:
        if not self.session_controller:
            return
        if self.session_controller.is_running():
            self.session_controller.stop()
            self.statusBar().showMessage("已停止 session")
            self.validate_current_routes()
            return

        route = self.audio_routing_page.selected_audio_routes()
        mode = self.audio_routing_page.selected_mode()
        try:
            self._sync_ui_to_config()
            self._build_pipelines_from_config()
            self._update_model_runtime_status()
            self._save_config_to_disk()
            result = self.session_controller.start(mode, route, sample_rate=self.config.sample_rate)
            if not result.ok:
                raise ValueError(result.message)
            self.statusBar().showMessage(result.message)
            self.validate_current_routes()
        except Exception as exc:
            self._report_error(f"start_session failed: {exc}")
            QMessageBox.critical(self, "啟動失敗", str(exc))

    def start_remote_capture(self, device_name: str) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.statusBar().showMessage("session 執行中，請先按「停止」")
            return
        try:
            self.remote_capture.start(device_name, sample_rate=self.config.sample_rate)
            self.statusBar().showMessage(f"開始擷取 remote_in: {device_name}")
        except Exception as exc:
            self._report_error(f"start_remote_capture failed: {exc}")
            QMessageBox.critical(self, "啟動失敗", str(exc))

    def stop_remote_capture(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.statusBar().showMessage("session 執行中，請先按「停止」")
            return
        self.remote_capture.stop()
        self.statusBar().showMessage("已停止 remote_in 擷取")

    def start_local_capture(self, device_name: str) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.statusBar().showMessage("session 執行中，請先按「停止」")
            return
        try:
            self.local_pipeline.start(device_name, sample_rate=self.config.sample_rate)
            self.statusBar().showMessage(f"開始 local pipeline: {device_name}")
        except Exception as exc:
            self._report_error(f"start_local_capture failed: {exc}")
            QMessageBox.critical(self, "啟動失敗", str(exc))

    def stop_local_capture(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.statusBar().showMessage("session 執行中，請先按「停止」")
            return
        self.local_pipeline.stop()
        self.statusBar().showMessage("已停止 local pipeline")

    def set_local_mute(self, muted: bool) -> None:
        self.local_capture.set_muted(muted)
        self.statusBar().showMessage("local_mic 已靜音" if muted else "local_mic 已解除靜音")

    def rebind_local_capture(self) -> None:
        try:
            self.local_capture.rebind()
            self.statusBar().showMessage("local_mic 重新綁定成功")
        except Exception as exc:
            self._report_error(f"rebind_local_capture failed: {exc}")
            QMessageBox.critical(self, "重新綁定失敗", str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._sync_ui_to_config()
            self._save_config_to_disk()
        except Exception as exc:
            self._report_error(f"save_on_close failed: {exc}")
        if self.session_controller:
            self.session_controller.stop()
        self.remote_capture.stop()
        self.local_capture.stop()
        self.audio_playback.stop()
        self.meeting_audio_playback.stop()
        super().closeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_window_to_screen)

    def _set_initial_window_geometry(self) -> None:
        default_width = 1140
        default_height = 780
        screen = self._current_screen()
        if screen is None:
            self.resize(default_width, default_height)
            return

        available = screen.availableGeometry()
        horizontal_margin = 48
        vertical_margin = 72
        max_width = max(360, available.width() - horizontal_margin)
        max_height = max(300, available.height() - vertical_margin)
        width = min(default_width, max_width)
        height = min(default_height, max_height)
        x = available.x() + (available.width() - width) // 2
        y = available.y() + (available.height() - height) // 2
        self.setGeometry(x, y, width, height)

    @staticmethod
    def _wrap_in_scroll_area(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _current_screen(self):
        if self.windowHandle() and self.windowHandle().screen():
            return self.windowHandle().screen()
        return QGuiApplication.primaryScreen()

    def _fit_window_to_screen(self) -> None:
        screen = self._current_screen()
        if screen is None:
            return

        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame_extra_w = max(0, frame.width() - self.width())
        frame_extra_h = max(0, frame.height() - self.height())
        border_margin = 16
        max_client_w = max(360, available.width() - border_margin - frame_extra_w)
        max_client_h = max(300, available.height() - border_margin - frame_extra_h)

        self.setMaximumSize(max_client_w, max_client_h)
        self.resize(min(self.width(), max_client_w), min(self.height(), max_client_h))

        frame = self.frameGeometry()
        max_x = available.x() + max(0, available.width() - frame.width())
        max_y = available.y() + max(0, available.height() - frame.height())
        x = min(max(frame.x(), available.x()), max_x)
        y = min(max(frame.y(), available.y()), max_y)
        self.move(x, y)

    def refresh_live_caption(self) -> None:
        self._check_provider_test_timeout()
        self._drain_provider_test_results()
        self.debug_panel.update_recent_errors(self._get_recent_errors())
        remote_original_items = self.transcript_buffer.latest("remote_original", limit=20)
        remote_translated_items = self.transcript_buffer.latest("remote_translated", limit=20)
        local_original_items = self.transcript_buffer.latest("local_original", limit=20)
        local_translated_items = self.transcript_buffer.latest("local_translated", limit=20)
        remote_original_lines = [self._format_transcript_line(item.text, item.is_final) for item in remote_original_items]
        remote_translated_lines = [self._format_transcript_line(item.text, item.is_final) for item in remote_translated_items]
        local_original_lines = [self._format_transcript_line(item.text, item.is_final) for item in local_original_items]
        local_translated_lines = [self._format_transcript_line(item.text, item.is_final) for item in local_translated_items]

        if remote_original_lines != self._remote_line_cache:
            self.live_caption_page.set_remote_original_lines(remote_original_lines)
            self._remote_line_cache = remote_original_lines

        if remote_translated_lines != self._remote_translated_line_cache:
            self.live_caption_page.set_remote_translated_lines(remote_translated_lines)
            self._remote_translated_line_cache = remote_translated_lines

        if local_original_lines != self._local_line_cache:
            self.live_caption_page.set_local_original_lines(local_original_lines)
            self._local_line_cache = local_original_lines

        if local_translated_lines != self._local_translated_line_cache:
            self.live_caption_page.set_local_translated_lines(local_translated_lines)
            self._local_translated_line_cache = local_translated_lines

    def _on_remote_original_transcript(self, _text: str) -> None:
        return

    def _on_remote_translated_transcript(self, _text: str) -> None:
        return

    def _on_local_original_transcript(self, _text: str) -> None:
        return

    def _on_local_translated_transcript(self, _text: str) -> None:
        return

    def _get_local_tts_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().local_tts_out

    def _get_meeting_tts_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().meeting_tts_out

    def test_meeting_tts_output(self) -> None:
        device_name = self._get_meeting_tts_output_device()
        try:
            audio = self.test_tts_provider.synthesize("Hello this is a meeting output test")
            self.meeting_audio_playback.play(audio=audio, sample_rate=24000, output_device_name=device_name)
            self.statusBar().showMessage(f"已測試英文送出: {device_name}")
        except Exception as exc:
            self._report_error(f"test_meeting_tts_output failed: {exc}")
            QMessageBox.critical(self, "測試失敗", str(exc))

    @staticmethod
    def _format_transcript_line(text: str, is_final: bool) -> str:
        state = "final" if is_final else "partial"
        return f"[{state}] {text}"

    def _report_error(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._error_lock:
            self.recent_errors.append(f"{timestamp} {message}")

    def export_diagnostics(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"diagnostics_{now}.txt")
        routes = self.audio_routing_page.selected_audio_routes()
        remote_stats = self.remote_capture.stats()
        local_stats = self.local_capture.stats()
        content = "\n".join(
            [
                "SyncTranslate Diagnostics",
                f"timestamp: {datetime.now().isoformat()}",
                f"config_path: {self.config_path}",
                f"asr_provider: {self.config.model.asr_provider}",
                f"translate_provider: {self.config.model.translate_provider}",
                f"tts_provider: {self.config.model.tts_provider}",
                f"openai_asr_model: {self.config.openai.asr_model}",
                f"openai_translate_model: {self.config.openai.translate_model}",
                f"openai_tts_model: {self.config.openai.tts_model}",
                f"openai_tts_voice: {self.config.openai.tts_voice}",
                f"remote_in: {routes.remote_in}",
                f"local_mic_in: {routes.local_mic_in}",
                f"local_tts_out: {routes.local_tts_out}",
                f"meeting_tts_out: {routes.meeting_tts_out}",
                f"remote_running: {remote_stats.running}",
                f"remote_level: {remote_stats.level:.4f}",
                f"local_running: {local_stats.running}",
                f"local_level: {local_stats.level:.4f}",
                "provider_test_last_success:",
                *self._provider_test_last_success_lines(),
                "recent_errors:",
                *self._get_recent_errors()[-20:],
            ]
        )
        output_path.write_text(content, encoding="utf-8")
        self.statusBar().showMessage(f"已匯出診斷資訊: {output_path}")

    def test_asr_provider(self) -> None:
        self._start_provider_test("ASR", self._run_test_asr_provider)

    def test_translate_provider(self) -> None:
        self._start_provider_test("Translate", self._run_test_translate_provider)

    def test_tts_provider_call(self) -> None:
        self._start_provider_test("TTS", self._run_test_tts_provider)

    def _build_pipelines_from_config(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.session_controller.stop()

        try:
            remote_asr_provider = create_asr_provider(
                provider_name=self.config.model.asr_provider,
                language=self.config.language.remote_source,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.asr_model,
            )
            remote_translate_provider = create_translate_provider(
                provider_name=self.config.model.translate_provider,
                source_lang=self.config.language.remote_source,
                target_lang=self.config.language.remote_target,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.translate_model,
            )
            remote_tts_provider = create_tts_provider(
                provider_name=self.config.model.tts_provider,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.tts_model,
                openai_voice=self.config.openai.tts_voice,
            )
            local_asr_provider = create_asr_provider(
                provider_name=self.config.model.asr_provider,
                language=self.config.language.local_source,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.asr_model,
            )
            local_translate_provider = create_translate_provider(
                provider_name=self.config.model.translate_provider,
                source_lang=self.config.language.local_source,
                target_lang=self.config.language.local_target,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.translate_model,
            )
            local_tts_provider = create_tts_provider(
                provider_name=self.config.model.tts_provider,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.tts_model,
                openai_voice=self.config.openai.tts_voice,
            )
            self.test_tts_provider = create_tts_provider(
                provider_name=self.config.model.tts_provider,
                openai_api_key_env=self.config.openai.api_key_env,
                openai_base_url=self.config.openai.base_url,
                openai_model=self.config.openai.tts_model,
                openai_voice=self.config.openai.tts_voice,
            )
        except Exception as exc:
            self._report_error(f"model provider build failed, fallback to mock: {exc}")
            remote_asr_provider = create_asr_provider(provider_name="mock", language=self.config.language.remote_source)
            remote_translate_provider = create_translate_provider(
                provider_name="mock",
                source_lang=self.config.language.remote_source,
                target_lang=self.config.language.remote_target,
            )
            remote_tts_provider = create_tts_provider(provider_name="mock")
            local_asr_provider = create_asr_provider(provider_name="mock", language=self.config.language.local_source)
            local_translate_provider = create_translate_provider(
                provider_name="mock",
                source_lang=self.config.language.local_source,
                target_lang=self.config.language.local_target,
            )
            local_tts_provider = create_tts_provider(provider_name="mock")
            self.test_tts_provider = create_tts_provider(provider_name="mock")

        self.remote_pipeline = RemotePipeline(
            audio_capture=self.remote_capture,
            transcript_buffer=self.transcript_buffer,
            audio_playback=self.audio_playback,
            asr_provider=remote_asr_provider,
            translate_provider=remote_translate_provider,
            tts_provider=remote_tts_provider,
            get_local_tts_output_device=self._get_local_tts_output_device,
            on_original_transcript=self._on_remote_original_transcript,
            on_translated_transcript=self._on_remote_translated_transcript,
            on_error=self._report_error,
        )
        self.local_pipeline = LocalPipeline(
            audio_capture=self.local_capture,
            transcript_buffer=self.transcript_buffer,
            audio_playback=self.meeting_audio_playback,
            asr_provider=local_asr_provider,
            translate_provider=local_translate_provider,
            tts_provider=local_tts_provider,
            get_meeting_tts_output_device=self._get_meeting_tts_output_device,
            on_original_transcript=self._on_local_original_transcript,
            on_translated_transcript=self._on_local_translated_transcript,
            on_error=self._report_error,
        )
        self.session_controller = SessionController(self.remote_pipeline, self.local_pipeline)
        self._update_model_runtime_status()

    def _get_recent_errors(self) -> list[str]:
        with self._error_lock:
            return list(self.recent_errors)

    def _update_model_runtime_status(self) -> None:
        uses_api_key_provider = any(
            provider in OPENAI_COMPATIBLE_PROVIDERS
            for provider in (
                self.config.model.asr_provider,
                self.config.model.translate_provider,
                self.config.model.tts_provider,
            )
        )
        if not uses_api_key_provider:
            if self.config.model.asr_provider == "local" or self.config.model.tts_provider == "edge_tts":
                self.models_page.set_runtime_status("Runtime status: local/edge providers ready (no OpenAI key needed)")
            else:
                self.models_page.set_runtime_status("Runtime status: mock providers ready")
            self._refresh_last_success_label()
            return

        env_name = self.config.openai.api_key_env
        has_key = bool(get_env_var(env_name, "").strip())
        if has_key:
            self.models_page.set_runtime_status(f"Runtime status: API key ready ({env_name} found)")
        else:
            self.models_page.set_runtime_status(f"Runtime status: WARNING - {env_name} not found")
        self._refresh_last_success_label()

    def _on_models_settings_changed(self) -> None:
        self.models_page.update_config(self.config)
        self._update_model_runtime_status()

    def _sync_ui_to_config(self) -> None:
        self.config.audio = self.audio_routing_page.selected_audio_routes()
        self.config.session_mode = self.audio_routing_page.selected_mode()
        self.models_page.update_config(self.config)

    def _save_config_to_disk(self) -> Path:
        return save_config(self.config, self.config_path)

    def _start_provider_test(self, test_name: str, runner: Callable[[], str]) -> None:
        if self._provider_test_running:
            self.statusBar().showMessage("已有測試進行中，請稍候")
            return
        self.models_page.update_config(self.config)
        self._provider_test_token += 1
        token = self._provider_test_token
        self._provider_test_active_token = token
        self._provider_test_running = True
        self._provider_test_started_at = time.monotonic()
        self.models_page.set_test_running(True)
        self.models_page.set_test_status(f"Provider test: {test_name} RUNNING...")

        def _worker() -> None:
            try:
                message = runner()
                self._provider_test_queue.put((token, test_name, True, message))
            except Exception as exc:
                self._provider_test_queue.put((token, test_name, False, str(exc)))

        Thread(target=_worker, daemon=True).start()

    def _drain_provider_test_results(self) -> None:
        while True:
            try:
                token, test_name, ok, message = self._provider_test_queue.get_nowait()
            except Empty:
                break

            if token != self._provider_test_active_token:
                continue
            self._provider_test_running = False
            self._provider_test_active_token = None
            self.models_page.set_test_running(False)
            if ok:
                self.models_page.set_test_status(f"Provider test: {test_name} OK ({message[:120]})")
                self.statusBar().showMessage(f"{test_name} provider 測試成功")
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._provider_test_last_success[test_name] = f"{stamp} {message}"
                self._refresh_last_success_label()
                self._persist_provider_test_state()
            else:
                self._report_error(f"test_{test_name.lower()}_provider failed: {message}")
                self.models_page.set_test_status(f"Provider test: {test_name} FAILED ({message})")
                QMessageBox.critical(self, f"{test_name} 測試失敗", message)

    def cancel_provider_test(self) -> None:
        if not self._provider_test_running:
            return
        self._provider_test_running = False
        self._provider_test_active_token = None
        self.models_page.set_test_running(False)
        self.models_page.set_test_status("Provider test: CANCELLED")
        self.statusBar().showMessage("已取消目前測試")

    def _check_provider_test_timeout(self) -> None:
        if not self._provider_test_running:
            return
        elapsed = time.monotonic() - self._provider_test_started_at
        if elapsed < self._provider_test_timeout_sec:
            return
        self._provider_test_running = False
        self._provider_test_active_token = None
        self.models_page.set_test_running(False)
        message = f"timeout after {int(self._provider_test_timeout_sec)}s"
        self.models_page.set_test_status(f"Provider test: FAILED ({message})")
        self._report_error(f"provider_test timeout: {message}")
        self.statusBar().showMessage("Provider 測試逾時")

    def _run_test_asr_provider(self) -> str:
        provider = create_asr_provider(
            provider_name=self.config.model.asr_provider,
            language=self.config.language.remote_source,
            openai_api_key_env=self.config.openai.api_key_env,
            openai_base_url=self.config.openai.base_url,
            openai_model=self.config.openai.asr_model,
        )
        audio, sample_rate = self._build_asr_test_audio()
        text = provider.final_text(audio=audio, sample_rate=sample_rate, segment_index=1)
        if not text.strip():
            raise ValueError("ASR provider returned empty text.")
        return text

    def _run_test_translate_provider(self) -> str:
        provider = create_translate_provider(
            provider_name=self.config.model.translate_provider,
            source_lang=self.config.language.remote_source,
            target_lang=self.config.language.remote_target,
            openai_api_key_env=self.config.openai.api_key_env,
            openai_base_url=self.config.openai.base_url,
            openai_model=self.config.openai.translate_model,
        )
        output = provider.translate("Hello, this is a translation test.")
        if not output.strip():
            raise ValueError("Translate provider returned empty text.")
        return output

    def _run_test_tts_provider(self) -> str:
        provider = create_tts_provider(
            provider_name=self.config.model.tts_provider,
            openai_api_key_env=self.config.openai.api_key_env,
            openai_base_url=self.config.openai.base_url,
            openai_model=self.config.openai.tts_model,
            openai_voice=self.config.openai.tts_voice,
        )
        audio = provider.synthesize("Hello this is a TTS provider test.")
        if audio.size == 0:
            raise ValueError("TTS provider returned empty audio.")
        return f"samples={audio.shape[0]}"

    def _refresh_last_success_label(self) -> None:
        if not self._provider_test_last_success:
            self.models_page.set_last_success("Last success: -")
            return
        parts = []
        for key in ("ASR", "Translate", "TTS"):
            if key in self._provider_test_last_success:
                preview = self._truncate_preview(self._provider_test_last_success[key])
                parts.append(f"{key}: {preview}")
        self.models_page.set_last_success("Last success: " + " | ".join(parts))

    def clear_provider_test_state(self) -> None:
        self.cancel_provider_test()
        self._provider_test_last_success.clear()
        self.models_page.set_test_status("Provider test: -")
        self._refresh_last_success_label()
        self._persist_provider_test_state()
        self.statusBar().showMessage("已清除 provider 測試狀態")

    def _provider_test_last_success_lines(self) -> list[str]:
        lines: list[str] = []
        for key in ("ASR", "Translate", "TTS"):
            value = self._provider_test_last_success.get(key, "-")
            lines.append(f"{key}: {value}")
        return lines

    def _persist_provider_test_state(self) -> None:
        self._sync_ui_to_config()
        self.config.provider_test_last_success.asr = self._provider_test_last_success.get("ASR", "")
        self.config.provider_test_last_success.translate = self._provider_test_last_success.get("Translate", "")
        self.config.provider_test_last_success.tts = self._provider_test_last_success.get("TTS", "")
        try:
            self._save_config_to_disk()
        except Exception as exc:
            self._report_error(f"persist_provider_test_state failed: {exc}")

    def _load_provider_test_state_from_config(self) -> None:
        mapping = {
            "ASR": self.config.provider_test_last_success.asr,
            "Translate": self.config.provider_test_last_success.translate,
            "TTS": self.config.provider_test_last_success.tts,
        }
        self._provider_test_last_success = {k: v for k, v in mapping.items() if v}

    def _truncate_preview(self, text: str) -> str:
        if len(text) <= self._last_success_preview_chars:
            return text
        return text[: self._last_success_preview_chars - 1] + "..."

    def _build_asr_test_audio(self) -> tuple[np.ndarray, int]:
        sample_rate = 24000
        try:
            edge_provider = create_tts_provider(
                provider_name="edge_tts",
                openai_voice="en-US-AvaMultilingualNeural",
            )
            audio = edge_provider.synthesize("Hello, this is an ASR provider test.", sample_rate=sample_rate)
            if audio.size > 0:
                return audio, sample_rate
        except Exception:
            pass

        fallback_rate = 16000
        t = np.linspace(0.0, 1.2, int(fallback_rate * 1.2), endpoint=False)
        tone = 0.15 * np.sin(2 * np.pi * 220 * t)
        return tone.reshape(-1, 1).astype(np.float32), fallback_rate
