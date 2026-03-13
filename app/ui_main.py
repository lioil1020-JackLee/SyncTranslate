from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock
from threading import Thread
import time
import tempfile

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QIcon, QShowEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget
import numpy as np

try:
    from opencc import OpenCC  # type: ignore
except Exception:
    OpenCC = None

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.app_bootstrap import build_pipeline_bundle
from app.audio_router import AudioRouter
from app.config_apply_service import ConfigApplyService
from app.debug_panel import DebugPanel
from app.diagnostics_service import export_runtime_diagnostics, export_session_report
from app.device_manager import DeviceManager
from app.events import ErrorEvent
from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.healthcheck import run_local_healthcheck
from app.local_ai.llm_provider import create_translation_provider
from app.local_ai.tts_factory import create_tts_engine
from app.pages_audio_routing import AudioRoutingPage
from app.pages_diagnostics import DiagnosticsPage
from app.pages_io_control import IoControlPage
from app.pages_live_caption import LiveCaptionPage
from app.pages_local_ai import LocalAiPage
from app.runtime_facade import RuntimeFacade
from app.schemas import AppConfig, TtsConfig
from app.session_controller import SessionController
from app.settings import load_config, save_config
from app.transcript_buffer import TranscriptBuffer


if OpenCC is not None:
    _S2T_CONVERTER = OpenCC("s2twp")
else:
    _S2T_CONVERTER = None


class MainWindow(QMainWindow):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self._window_geometry_ready = False
        self._apply_standard_window_flags()
        self.config_path = config_path
        self.config: AppConfig = load_config(self.config_path)
        self.device_manager = DeviceManager()
        self.meeting_capture = AudioCapture()
        self.local_capture = AudioCapture()
        self.speaker_playback = AudioPlayback()
        self.meeting_playback = AudioPlayback()
        self.transcript_buffer = TranscriptBuffer(max_items=300)
        self.recent_errors: list[str] = []
        self._error_lock = Lock()
        self.audio_router: AudioRouter | None = None
        self.session_controller: SessionController | None = None
        self._health_check_running = False
        self._health_check_queue: Queue[tuple[bool, object]] = Queue()
        self._health_check_started_at = 0.0
        self._health_check_timeout_sec = 45.0
        self._session_action_running = False
        self._session_action_name = ""
        self._session_action_queue: Queue[tuple[str, bool, object]] = Queue()
        self._pending_live_apply = False

        self.setWindowTitle("SyncTranslate - Local AI Runtime")
        self._set_window_icon()

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setStyleSheet(
            """
            QTabBar::tab {
                min-width: 170px;
                min-height: 40px;
                font-size: 12pt;
                padding: 6px 12px;
                color: #ddd;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                margin: 4px 6px;
            }
            QTabBar::tab:hover {
                background: rgba(255,255,255,0.03);
            }
            QTabBar::tab:!selected {
                color: #cfcfcf;
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.06);
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2b78d6, stop:1 #1a5fb0);
                color: white;
                border: 1px solid rgba(0,0,0,0.12);
            }
            """
        )
        self.audio_routing_page = AudioRoutingPage(
            on_route_changed=self._on_audio_routing_changed,
        )
        self.live_caption_page = LiveCaptionPage(
            on_clear_clicked=self.clear_live_caption,
            on_start_clicked=self.start_session,
            on_test_local_tts_clicked=self.test_local_tts_output,
            on_settings_changed=self._on_live_caption_settings_changed,
        )
        self.local_ai_page = LocalAiPage(
            on_settings_changed=self._on_local_ai_changed,
            on_health_check=self.run_health_check,
        )
        self.diagnostics_page = DiagnosticsPage(
            on_health_check=self.run_health_check,
            on_export_diagnostics=self.export_diagnostics,
            on_save_config=self.persist_config,
            on_reload_config=self.reload_config,
        )
        self.io_control_page = IoControlPage(self.audio_routing_page, self.diagnostics_page)
        self.debug_panel = DebugPanel()
        self.diagnostics_page.set_extra_panel(self.debug_panel)

        self.tabs.addTab(self._wrap_in_scroll_area(self.io_control_page), "音訊路由與診斷")
        self.tabs.addTab(self._wrap_in_scroll_area(self.live_caption_page), "即時字幕")
        self.tabs.addTab(self.local_ai_page, "參數設定")
        self._config_apply_service = ConfigApplyService(
            meeting_capture=self.meeting_capture,
            local_capture=self.local_capture,
            speaker_playback=self.speaker_playback,
            meeting_playback=self.meeting_playback,
            audio_routing_page=self.audio_routing_page,
            live_caption_page=self.live_caption_page,
            local_ai_page=self.local_ai_page,
        )
        self._runtime_facade = RuntimeFacade(self._create_pipeline_bundle)
        self.setCentralWidget(self.tabs)
        self._apply_content_minimum_height()
        # Now that pages are constructed and minimum heights calculated,
        # set initial window geometry based on the parameters page size.
        self._set_initial_window_geometry()
        self._build_pipelines_from_config()

        self.input_device_names: set[str] = set()
        self.output_device_names: set[str] = set()
        self._remote_line_cache: list[str] = []
        self._remote_translated_line_cache: list[str] = []
        self._local_line_cache: list[str] = []
        self._local_translated_line_cache: list[str] = []
        self._live_apply_ready = False
        self._suspend_live_apply = False
        self._live_apply_timer = QTimer(self)
        self._live_apply_timer.setSingleShot(True)
        self._live_apply_timer.setInterval(150)
        self._live_apply_timer.timeout.connect(self._apply_live_config_now)

        self.caption_timer = QTimer(self)
        self.caption_timer.setInterval(300)
        self.caption_timer.timeout.connect(self.refresh_live_caption)
        self.caption_timer.start()
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(120)
        self.health_timer.timeout.connect(self._drain_health_check_results)
        self.health_timer.start()
        self.session_timer = QTimer(self)
        self.session_timer.setInterval(120)
        self.session_timer.timeout.connect(self._drain_session_results)
        self.session_timer.start()
        self.refresh_from_system()
        self._live_apply_ready = True

    def refresh_from_system(self) -> None:
        input_devices = self.device_manager.list_input_devices()
        output_devices = self.device_manager.list_output_devices()
        self.input_device_names = {d.name for d in input_devices}
        self.output_device_names = {d.name for d in output_devices}

        self._suspend_live_apply = True
        try:
            self.audio_routing_page.set_devices(input_devices, output_devices)
            self.audio_routing_page.apply_config(self.config)
            self.live_caption_page.apply_config(self.config)
            self.local_ai_page.apply_config(self.config)
            self._apply_audio_route_levels()
        finally:
            self._suspend_live_apply = False

        self.statusBar().showMessage(
            f"Config: {self.config_path} | input={len(input_devices)} output={len(output_devices)}"
        )
        self.validate_current_routes()

    def persist_config(self) -> None:
        try:
            self._apply_live_config_now()
            saved_path = self._save_config_to_disk()
            self.statusBar().showMessage(f"設定已儲存: {saved_path}")
        except Exception as exc:
            self._report_error(f"persist_config failed: {exc}")
            self.statusBar().showMessage(f"設定儲存失敗: {exc}")

    def reload_config(self) -> None:
        if self.session_controller:
            self.session_controller.stop()
        self.config = load_config(self.config_path)
        self._runtime_facade.mark_dirty()
        self._ensure_pipelines_ready()
        self.refresh_from_system()
        self.statusBar().showMessage(f"已重載設定: {self.config_path}")

    def validate_current_routes(self) -> None:
        self.debug_panel.update_recent_errors(self._get_recent_errors())
        is_running = self.session_controller.is_running() if self.session_controller else False
        self.live_caption_page.set_start_enabled(True)
        self.live_caption_page.set_start_label("停止" if is_running else "開始")
        self.live_caption_page.set_direction_controls_enabled(not is_running and not self._session_action_running)
        self._update_live_panel_statuses(
            remote_original_active=bool(self._remote_line_cache),
            remote_translated_active=bool(self._remote_translated_line_cache),
            local_original_active=bool(self._local_line_cache),
            local_translated_active=bool(self._local_translated_line_cache),
        )
    def _on_audio_routing_changed(self) -> None:
        self.validate_current_routes()
        self._apply_audio_route_levels_from_ui()
        self._schedule_live_apply()

    def _on_local_ai_changed(self) -> None:
        self._schedule_live_apply()

    def _on_live_caption_settings_changed(self) -> None:
        self._schedule_live_apply()

    def _schedule_live_apply(self) -> None:
        if not self._live_apply_ready or self._suspend_live_apply:
            return
        if self._session_action_running:
            self._pending_live_apply = True
            self.statusBar().showMessage("設定變更已暫存，等待目前的啟動/停止動作完成後套用")
            return
        if self.session_controller and self.session_controller.is_running():
            self._pending_live_apply = True
            self.statusBar().showMessage("設定變更已暫存，請先停止 session，程式會在停止後自動套用")
            return
        self._live_apply_timer.start()

    def _apply_live_config_now(self) -> None:
        if not self._live_apply_ready:
            return
        if self._session_action_running:
            self._pending_live_apply = True
            return
        if self._live_apply_timer.isActive():
            self._live_apply_timer.stop()

        was_running = self.session_controller.is_running() if self.session_controller else False
        if was_running:
            self._pending_live_apply = True
            self.statusBar().showMessage("設定變更已暫存，請先停止 session，程式會在停止後自動套用")
            return
        route = self.audio_routing_page.selected_audio_routes()
        mode = self.live_caption_page.selected_mode()
        try:
            self._pending_live_apply = False
            self._sync_ui_to_config()
            self._runtime_facade.mark_dirty()
            # Live apply should not write config.yaml automatically.
            # Keep config changes in memory; only explicit save persists to disk.
            path = Path(self.config_path)

            if was_running and self.session_controller:
                self.statusBar().showMessage(f"設定已暫存: {path}")
            else:
                self._ensure_pipelines_ready()
                self.statusBar().showMessage(f"設定已套用(未儲存): {path}")
            self.validate_current_routes()
        except Exception as exc:
            self._report_error(f"auto_apply_config failed: {exc}")
            self.statusBar().showMessage(f"設定套用失敗: {exc}")

    def start_session(self) -> None:
        if self._session_action_running:
            return
        self._ensure_pipelines_ready()
        if not self.session_controller:
            return
        if self.session_controller.is_running():
            self._run_session_action("stop")
            return

        route = self.audio_routing_page.selected_audio_routes()
        mode = self.live_caption_page.selected_mode()
        try:
            self._sync_ui_to_config()
            self._runtime_facade.mark_dirty()
            self._ensure_pipelines_ready()
            self.clear_live_caption()
            self._run_session_action(
                "start",
                mode=mode,
                route=route,
                sample_rate=self.config.runtime.sample_rate,
                chunk_ms=self.config.runtime.chunk_ms,
            )
        except Exception as exc:
            self._report_error(f"start_session failed: {exc}")
            QMessageBox.critical(self, "啟動失敗", str(exc))

    def clear_live_caption(self) -> None:
        self.live_caption_page.clear()
        self.transcript_buffer.clear()
        self._remote_line_cache = []
        self._remote_translated_line_cache = []
        self._local_line_cache = []
        self._local_translated_line_cache = []
        self._update_live_panel_statuses(
            remote_original_active=False,
            remote_translated_active=False,
            local_original_active=False,
            local_translated_active=False,
        )
        self.statusBar().showMessage("字幕已清空")

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._sync_ui_to_config()
        except Exception as exc:
            self._report_error(f"sync_on_close failed: {exc}")
        if self.session_controller:
            self.session_controller.stop()
        self.meeting_capture.stop()
        self.local_capture.stop()
        self.speaker_playback.stop()
        self.meeting_playback.stop()
        super().closeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._window_geometry_ready:
            return
        self._window_geometry_ready = True
        self._ensure_window_decorations()
        QTimer.singleShot(0, self._fit_window_to_screen)

    def _set_initial_window_geometry(self) -> None:
        default_width = 1180
        default_height = 980
        screen = self._current_screen()
        if screen is None:
            self.resize(default_width, default_height)
            return
        available = screen.availableGeometry()
        horizontal_margin = 48
        max_width = max(360, available.width() - horizontal_margin)
        width = min(default_width, max_width)
        preferred_client_h = self._preferred_client_height()
        client_h = min(preferred_client_h, max(640, available.height() - 72))
        x = available.x() + (available.width() - width) // 2
        y = available.y()
        self.resize(width, client_h)
        self.move(x, y)

    def _preferred_client_height(self) -> int:
        page_height = 0
        if hasattr(self, "local_ai_page") and self.local_ai_page is not None:
            try:
                page_height = max(
                    self.local_ai_page.sizeHint().height(),
                    self.local_ai_page.minimumSizeHint().height(),
                )
            except Exception:
                page_height = self.local_ai_page.minimumHeight()
        tab_bar_h = self.tabs.tabBar().sizeHint().height() if hasattr(self, "tabs") else 0
        status_h = self.statusBar().sizeHint().height() if self.statusBar() else 0
        return max(760, min(1020, page_height + tab_bar_h + status_h + 12))

    def _standard_window_flags(self):
        return (
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

    def _apply_standard_window_flags(self) -> None:
        try:
            self.setWindowFlag(Qt.FramelessWindowHint, False)
        except Exception:
            pass
        try:
            self.setWindowFlag(Qt.CustomizeWindowHint, False)
        except Exception:
            pass
        self.setWindowFlags(self._standard_window_flags())

    def _set_window_icon(self) -> None:
        icon = QIcon("lioil.ico")
        if not icon.isNull():
            self.setWindowIcon(icon)

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

    def _ensure_window_decorations(self) -> None:
        try:
            self._apply_standard_window_flags()
            self.showNormal()
        except Exception:
            pass

    def _apply_content_minimum_height(self) -> None:
        desired_height = max(760, self._preferred_client_height() - 28)
        self.setMinimumHeight(desired_height)

    def _fit_window_to_screen(self) -> None:
        screen = self._current_screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame_extra_w = max(0, frame.width() - self.width())
        frame_extra_h = max(0, frame.height() - self.height())
        max_client_w = max(360, available.width() - frame_extra_w)
        max_client_h = max(300, available.height() - frame_extra_h)
        target_w = min(self.width(), max_client_w)
        target_h = min(max(self.height(), self.minimumHeight()), max_client_h)
        self.resize(target_w, target_h)
        frame = self.frameGeometry()
        x = available.x() + max(0, (available.width() - frame.width()) // 2)
        max_y = available.y() + max(0, available.height() - frame.height())
        y = min(max(frame.y(), available.y()), max_y)
        self.move(x, y)

    def refresh_live_caption(self) -> None:
        self._drain_health_check_results()
        self.debug_panel.update_recent_errors(self._get_recent_errors())
        self.debug_panel.update_runtime_stats(self._build_runtime_stats_text())
        remote_original_items = self.transcript_buffer.latest("meeting_original", limit=20)
        remote_translated_items = self.transcript_buffer.latest("meeting_translated", limit=20)
        local_original_items = self.transcript_buffer.latest("local_original", limit=20)
        local_translated_items = self.transcript_buffer.latest("local_translated", limit=20)
        remote_original_lines = self._build_transcript_lines(remote_original_items)
        remote_translated_lines = self._build_transcript_lines(remote_translated_items)
        local_original_lines = self._build_transcript_lines(local_original_items)
        local_translated_lines = self._build_transcript_lines(local_translated_items)

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

        self._update_live_panel_statuses(
            remote_original_active=bool(remote_original_lines),
            remote_translated_active=bool(remote_translated_lines),
            local_original_active=bool(local_original_lines),
            local_translated_active=bool(local_translated_lines),
        )

    def _resolve_active_sources(self) -> set[str]:
        if self._session_action_running and self._session_action_name == "start":
            mode = self.live_caption_page.selected_mode()
            if mode == "meeting_to_local":
                return {"remote"}
            if mode == "local_to_meeting":
                return {"local"}
            if mode == "bidirectional":
                return {"local", "remote"}
            return set()

        if not self.session_controller or not self.session_controller.is_running() or not self.audio_router:
            return set()

        try:
            stats = self.audio_router.stats()
            return set(stats.active_sources)
        except Exception:
            return set()

    def _resolve_source_ready(self) -> dict[str, bool]:
        if not self.audio_router:
            return {"local": False, "remote": False}
        try:
            stats = self.audio_router.stats()
            ready: dict[str, bool] = {"local": False, "remote": False}
            for source in ("local", "remote"):
                capture = stats.capture.get(source) or {}
                running = bool(capture.get("running", False))
                frame_count = int(capture.get("frame_count", 0) or 0)
                ready[source] = running and frame_count > 0
            return ready
        except Exception:
            return {"local": False, "remote": False}

    def _update_live_panel_statuses(
        self,
        *,
        remote_original_active: bool,
        remote_translated_active: bool,
        local_original_active: bool,
        local_translated_active: bool,
    ) -> None:
        active_sources = self._resolve_active_sources()
        ready_sources = self._resolve_source_ready()
        is_preparing = self._session_action_running and self._session_action_name == "start"

        remote_enabled = "remote" in active_sources
        local_enabled = "local" in active_sources

        def _status(enabled: bool, source_ready: bool) -> str:
            if not enabled:
                return "idle"
            if is_preparing or not source_ready:
                return "preparing"
            # Once session is running and source is active, user can start speaking.
            return "running"

        self.live_caption_page.set_panel_statuses(
            remote_original=_status(remote_enabled, bool(ready_sources.get("remote", False))),
            remote_translated=_status(remote_enabled, bool(ready_sources.get("remote", False))),
            local_original=_status(local_enabled, bool(ready_sources.get("local", False))),
            local_translated=_status(local_enabled, bool(ready_sources.get("local", False))),
        )

    def _get_speaker_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().speaker_out

    def _get_meeting_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().meeting_out

    def test_local_tts_output(self) -> None:
        device_name = self._get_speaker_output_device()
        try:
            self._sync_ui_to_config()
            target_lang = self.config.language.local_target
            test_tts = self._tts_for_language_target(self.config.meeting_tts, target_lang)
            audio, sample_rate, engine_label = self._build_output_test_audio(
                primary_tts=test_tts,
                text=self._tts_test_text(target_lang),
            )
            self.speaker_playback.play(
                audio=audio,
                sample_rate=sample_rate,
                output_device_name=device_name,
                blocking=True,
            )
            self.statusBar().showMessage(f"本地輸出TTS測試完成: {device_name} ({engine_label})")
        except Exception as exc:
            self._report_error(f"test_local_tts_output failed: {exc}")
            QMessageBox.critical(self, "測試失敗", self._format_playback_error(exc))

    @staticmethod
    def _format_transcript_line(text: str, is_final: bool) -> str:
        if _S2T_CONVERTER is not None:
            try:
                text = _S2T_CONVERTER.convert(text)
            except Exception:
                pass
        state = "final" if is_final else "partial"
        return f"[{state}] {text}"

    @classmethod
    def _build_transcript_lines(cls, items) -> list[str]:
        return [cls._format_transcript_line(item.text, item.is_final) for item in items]

    def _report_error(self, message: str | ErrorEvent) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path("logs") / "runtime_events.log"
        log_path.parent.mkdir(exist_ok=True)
        text = message.to_log_line() if isinstance(message, ErrorEvent) else str(message)
        with self._error_lock:
            line = f"{timestamp} {text}"
            self.recent_errors.append(line)
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    def export_diagnostics(self) -> None:
        routes = self.audio_routing_page.selected_audio_routes()
        output_path = export_runtime_diagnostics(
            config_path=self.config_path,
            config=self.config,
            routes=routes,
            runtime_stats_text=self._build_runtime_stats_text(),
            recent_errors=self._get_recent_errors(),
        )
        self.statusBar().showMessage(f"已匯出診斷資訊: {output_path}")

    def _create_pipeline_bundle(self, config: AppConfig, pipeline_revision: int):
        return build_pipeline_bundle(
            config=config,
            pipeline_revision=pipeline_revision,
            transcript_buffer=self.transcript_buffer,
            local_capture=self.local_capture,
            meeting_capture=self.meeting_capture,
            speaker_playback=self.speaker_playback,
            meeting_playback=self.meeting_playback,
            get_local_output_device=self._get_speaker_output_device,
            get_remote_output_device=self._get_meeting_output_device,
            on_error=self._report_error,
            on_diagnostic_event=lambda message: self._report_error(f"[router] {message}"),
        )

    def _build_pipelines_from_config(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.session_controller.stop()
        bundle = self._runtime_facade.rebuild(self.config)
        self.audio_router = bundle.audio_router
        self.session_controller = bundle.session_controller

    def _ensure_pipelines_ready(self) -> None:
        bundle = self._runtime_facade.ensure_ready(self.config)
        self.audio_router = bundle.audio_router
        self.session_controller = bundle.session_controller

    def run_health_check(self, warmup: bool) -> None:
        if self._health_check_running:
            self.statusBar().showMessage("health check 進行中")
            return
        self._sync_ui_to_config()
        config_path = self._create_healthcheck_snapshot_config()
        self._health_check_running = True
        self._health_check_started_at = time.monotonic()
        self.local_ai_page.set_runtime_status("健康檢查進行中")
        self.diagnostics_page.set_health_details("健康檢查進行中...")
        self.statusBar().showMessage("健康檢查進行中...")

        def _worker() -> None:
            try:
                if getattr(sys, "frozen", False):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "--healthcheck-worker",
                            "--healthcheck-config",
                            config_path,
                            "--healthcheck-warmup",
                            "true" if warmup else "false",
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                else:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "app.local_ai.healthcheck_worker",
                            config_path,
                            "true" if warmup else "false",
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                if completed.returncode != 0:
                    stderr = (completed.stderr or "").strip()
                    stdout = (completed.stdout or "").strip()
                    detail = stderr or stdout or f"health check subprocess exited with code {completed.returncode}"
                    raise RuntimeError(detail)
                payload = (completed.stdout or "").strip()
                if not payload:
                    raise RuntimeError("health check subprocess returned no result")
                report = LocalHealthReport(**json.loads(payload))
                self._health_check_queue.put((True, report))
            except Exception as exc:
                self._health_check_queue.put((False, exc))
            finally:
                try:
                    os.unlink(config_path)
                except Exception:
                    pass

        Thread(target=_worker, daemon=True).start()

    def _create_healthcheck_snapshot_config(self) -> str:
        snapshot_dir = Path("logs") / "tmp"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="healthcheck_",
            dir=str(snapshot_dir),
            delete=False,
            encoding="utf-8",
        ) as fp:
            save_config(self.config, fp.name)
            return fp.name

    def _drain_health_check_results(self) -> None:
        if not self._health_check_running:
            return
        elapsed = time.monotonic() - self._health_check_started_at if self._health_check_started_at else 0.0
        if elapsed >= self._health_check_timeout_sec:
            self._health_check_running = False
            self._health_check_started_at = 0.0
            summary = "健康檢查：失敗"
            message = f"健康檢查逾時，超過 {int(self._health_check_timeout_sec)} 秒"
            self.local_ai_page.set_runtime_status(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(message)
            self._report_error(message)
            return
        try:
            ok, payload = self._health_check_queue.get_nowait()
        except Empty:
            return
        self._health_check_running = False
        self._health_check_started_at = 0.0
        if not ok:
            message = str(payload)
            summary = "健康檢查：失敗"
            self.local_ai_page.set_runtime_status(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(summary)
            self._report_error(f"health_check failed: {message}")
            return

        report = payload
        summary = "健康檢查：正常" if report.ok else "健康檢查：失敗"
        self.local_ai_page.set_runtime_status(summary)
        self.diagnostics_page.set_asr_details(f"{'正常' if report.asr_ok else '失敗'} - {report.asr_message}")
        self.diagnostics_page.set_llm_details(f"{'正常' if report.llm_ok else '失敗'} - {report.llm_message}")
        self.diagnostics_page.set_tts_details(f"{'正常' if report.tts_ok else '失敗'} - {report.tts_message}")
        self.statusBar().showMessage(summary)

    def _run_session_action(
        self,
        action: str,
        *,
        mode: str | None = None,
        route=None,
        sample_rate: int | None = None,
        chunk_ms: int | None = None,
    ) -> None:
        if not self.session_controller or self._session_action_running:
            return
        self._session_action_running = True
        self._session_action_name = action
        self.live_caption_page.set_start_enabled(False)
        if action == "stop":
            self.live_caption_page.set_start_label("停止中...")
            self.statusBar().showMessage("正在停止 session...")
        else:
            self.live_caption_page.set_start_label("啟動中...")
            self.statusBar().showMessage("正在啟動 session...")
            self._update_live_panel_statuses(
                remote_original_active=False,
                remote_translated_active=False,
                local_original_active=False,
                local_translated_active=False,
            )

        def _worker() -> None:
            try:
                if action == "stop":
                    result = self.session_controller.stop()
                else:
                    if mode is None or route is None or sample_rate is None:
                        raise ValueError("missing session start parameters")
                    result = self.session_controller.start(
                        mode,
                        route,
                        sample_rate=sample_rate,
                        chunk_ms=chunk_ms or self.config.runtime.chunk_ms,
                    )
                self._session_action_queue.put((action, True, result))
            except Exception as exc:
                self._session_action_queue.put((action, False, exc))

        Thread(target=_worker, daemon=True).start()

    def _drain_session_results(self) -> None:
        if not self._session_action_running:
            return
        try:
            action, ok, payload = self._session_action_queue.get_nowait()
        except Empty:
            return

        self._session_action_running = False
        self._session_action_name = ""
        if not ok:
            message = str(payload)
            self._report_error(f"session_{action} failed: {message}")
            self.statusBar().showMessage(f"session {action} failed")
            QMessageBox.critical(self, "Session 錯誤", message)
            self.validate_current_routes()
            return

        result = payload
        if not result.ok:
            self._report_error(f"session_{action} failed: {result.message}")
            self.statusBar().showMessage(result.message)
            if action == "start":
                QMessageBox.critical(self, "啟動失敗", result.message)
            self.validate_current_routes()
            return

        self.statusBar().showMessage(result.message)
        if action == "stop" and result.payload is not None:
            self._export_session_report(result.payload)
        self.validate_current_routes()
        if action == "stop" and self._pending_live_apply:
            QTimer.singleShot(0, self._apply_live_config_now)

    def _export_session_report(self, payload: dict[str, object]) -> None:
        try:
            routes = self.audio_routing_page.selected_audio_routes()
            export_session_report(
                config_path=self.config_path,
                config=self.config,
                routes=routes,
                payload=payload,
                recent_errors=self._get_recent_errors(),
            )
        except Exception as exc:
            self._report_error(f"export_session_report failed: {exc}")

    def _get_recent_errors(self) -> list[str]:
        with self._error_lock:
            return list(self.recent_errors)

    def _build_runtime_stats_text(self) -> str:
        lines = [
            f"saved meeting: {self.config.language.meeting_source} -> {self.config.language.meeting_target}",
            f"saved local: {self.config.language.local_source} -> {self.config.language.local_target}",
            f"meeting_tts: {self.config.meeting_tts.engine} voice={self.config.meeting_tts.voice_name or self.config.meeting_tts.model_path}",
            f"local_tts: {self.config.local_tts.engine} voice={self.config.local_tts.voice_name or self.config.local_tts.model_path}",
        ]
        if not self.audio_router:
            lines.append("router: not built")
            return "\n".join(lines)

        router_stats = self.audio_router.stats()
        lines.append(
            "router: "
            f"running={router_stats.running} "
            f"active={','.join(router_stats.active_sources) if router_stats.active_sources else '-'}"
        )
        state = router_stats.state
        lines.append(
            "state: "
            f"local_asr={state['local_asr_enabled']} remote_asr={state['remote_asr_enabled']} "
            f"local_tts_busy={state['local_tts_busy']} remote_tts_busy={state['remote_tts_busy']} "
            f"local_resume_in_ms={state['local_resume_in_ms']} "
            f"remote_resume_in_ms={state['remote_resume_in_ms']}"
        )
        for label, source in (("meeting", "remote"), ("local", "local")):
            capture = router_stats.capture[source]
            asr = router_stats.asr[source]
            lines.append(
                (
                    f"{label}: capture={capture['running']} "
                    f"rate={int(capture['sample_rate']) if capture['sample_rate'] else 0} "
                    f"frames={capture['frame_count']} level={capture['level']:.5f} "
                    f"queue={asr['queue_size']} dropped={asr['dropped_chunks']} "
                    f"partials={asr['partial_count']} finals={asr['final_count']} "
                    f"vad_rms={asr['vad_rms']:.5f} vad_th={asr['vad_threshold']:.5f}"
                )
            )
            if capture["last_error"]:
                lines.append(f"{label}: capture_error={capture['last_error']}")
            if asr["last_debug"]:
                lines.append(f"{label}: asr_last={asr['last_debug']}")
        tts = router_stats.tts
        lines.append(
            "tts: "
            f"depth={tts['queue_depth']} local_depth={tts['queue_depth_local']} remote_depth={tts['queue_depth_remote']} "
            f"drop_local={tts['drop_count_local']} drop_remote={tts['drop_count_remote']} "
            f"oldest_age_ms={tts['oldest_age_ms']:.1f}"
        )
        return "\n".join(lines)

    def _sync_ui_to_config(self) -> None:
        self._config_apply_service.sync_ui_to_config(self.config)

    def _apply_audio_route_levels(self) -> None:
        self._config_apply_service.apply_audio_route_levels(self.config.audio)

    def _apply_audio_route_levels_from_ui(self) -> None:
        self._config_apply_service.apply_audio_route_levels_from_ui()

    def _save_config_to_disk(self) -> Path:
        return save_config(self.config, self.config_path)

    def _build_output_test_audio(self, *, primary_tts: TtsConfig, text: str) -> tuple[np.ndarray, int, str]:
        try:
            tts = create_tts_engine(primary_tts)
            audio = tts.synthesize(text, sample_rate=primary_tts.sample_rate)
            if audio.size == 0:
                raise ValueError("tts returned empty audio")
            return audio, primary_tts.sample_rate, "configured tts"
        except Exception as exc:
            engine = (primary_tts.engine or "").strip() or "unknown"
            voice = (primary_tts.voice_name or "").strip() or primary_tts.model_path
            raise ValueError(f"Configured TTS failed ({engine}, {voice}): {exc}") from exc

    @staticmethod
    def _format_playback_error(exc: Exception) -> str:
        text = str(exc).strip()
        if "Unable to play TTS audio" in text:
            return "無法在選定的喇叭輸出播放測試音。請改選可用的喇叭或耳機裝置。\n\n詳細資訊:\n" + text
        if "No audio was received" in text:
            return "TTS 沒有回傳音訊。若你使用 Edge TTS，常見原因是聲線語系和測試文字語言不相容。\n\n詳細資訊:\n" + text
        return text

    @staticmethod
    def _tts_test_text(language: str) -> str:
        normalized = (language or "").lower()
        if normalized.startswith("zh"):
            return "這是翻譯語音測試。"
        if normalized.startswith("ja"):
            return "これは音声テストです。"
        return "This is a translated speech test."

    @classmethod
    def _tts_for_language_target(cls, base: TtsConfig, language: str) -> TtsConfig:
        normalized = (language or "").strip().lower()
        voice = (base.voice_name or "").strip()
        if cls._voice_matches_language(voice, normalized):
            return base
        fallback_voice = cls._default_voice_for_language(normalized)
        if not fallback_voice:
            return base
        return replace(base, voice_name=fallback_voice)

    @staticmethod
    def _voice_matches_language(voice_name: str, normalized_language: str) -> bool:
        lowered = (voice_name or "").strip().lower()
        if not lowered:
            return False
        if normalized_language.startswith("zh"):
            return lowered.startswith("zh-")
        if normalized_language.startswith("en"):
            return lowered.startswith("en-")
        if normalized_language.startswith("ja"):
            return lowered.startswith("ja-")
        return True

    @staticmethod
    def _default_voice_for_language(normalized_language: str) -> str:
        if normalized_language.startswith("zh"):
            return "zh-TW-HsiaoChenNeural"
        if normalized_language.startswith("en"):
            return "en-US-JennyNeural"
        if normalized_language.startswith("ja"):
            return "ja-JP-NanamiNeural"
        return ""

