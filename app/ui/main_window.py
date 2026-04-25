from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import sys
import time
from threading import Lock
from threading import Thread

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QIcon, QShowEvent
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QMainWindow, QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget

try:
    from opencc import OpenCC  # type: ignore
except Exception:
    OpenCC = None

from app.application.audio_router import AudioRouter
from app.application.export_service import ExportService
from app.application.healthcheck_service import HealthCheckService
from app.application.runtime_orchestrator import RuntimeFacade
from app.application.session_service import SessionController
from app.application.settings_service import SettingsService
from app.application.config_apply_service import ConfigApplyService
from app.infra.audio.device_volume_controller import SystemDeviceVolumeController
from app.bootstrap.dependency_container import build_pipeline_bundle
from app.domain.events import ErrorEvent
from app.infra.audio.capture import AudioCapture
from app.infra.audio.device_registry import DeviceManager, canonical_device_name
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AppConfig, translation_enabled_for_source
from app.application.transcript_service import TranscriptBuffer
from app.ui.pages.audio_routing_page import AudioRoutingPage
from app.ui.pages.diagnostics_page import DiagnosticsPage
from app.ui.pages.live_caption_page import LiveCaptionPage
from app.ui.pages.local_ai_page import LocalAiPage
from app.ui.pages.settings_page import SettingsPage
from app.bootstrap.runtime_paths import runtime_logs_dir


if OpenCC is not None:
    _S2T_CONVERTER = OpenCC("s2twp")
else:
    _S2T_CONVERTER = None


class MainWindow(QMainWindow):
    def __init__(self, config_path: str, device_volume_controller: SystemDeviceVolumeController | None = None) -> None:
        super().__init__()
        self._window_geometry_ready = False
        self._apply_standard_window_flags()
        self.config_path = config_path
        self._settings_service = SettingsService(self.config_path)
        self._export_service = ExportService()
        self._healthcheck_service = HealthCheckService(
            settings_service=self._settings_service,
            timeout_sec=45.0,
        )
        self.config: AppConfig = self._settings_service.load()
        self.device_manager = DeviceManager()
        self.meeting_capture = AudioCapture()
        self.local_capture = AudioCapture()
        self.speaker_playback = AudioPlayback()
        self.meeting_playback = AudioPlayback()
        self.transcript_buffer = TranscriptBuffer(max_items=None)
        self.recent_errors: list[str] = []
        self._error_lock = Lock()
        self._error_throttle_state: dict[str, dict[str, float | int]] = {}
        self._error_throttle_window_sec = 2.5
        self._asr_detected_lock = Lock()
        self._latest_asr_detected_lang: dict[str, str] = {"local": "", "remote": ""}
        self.audio_router: AudioRouter | None = None
        self.session_controller: SessionController | None = None
        self._session_action_running = False
        self._session_action_name = ""
        self._session_action_queue: Queue[tuple[str, bool, object]] = Queue()
        self._pending_live_apply = False
        self._pending_live_caption_apply = False
        self._live_apply_ready = False
        self._resolved_window_icon = QIcon()

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
            on_settings_changed=self._on_live_caption_settings_changed,
            on_output_mode_changed=self._on_output_mode_changed,
        )
        self.local_ai_page = LocalAiPage(
            on_settings_changed=self._on_local_ai_changed,
            on_health_check=self.run_system_check,
            on_save_config=self.persist_config,
        )
        self.diagnostics_page = DiagnosticsPage(
        )
        self.settings_page = SettingsPage(
            audio_routing_page=self.audio_routing_page,
            local_ai_page=self.local_ai_page,
            diagnostics_page=self.diagnostics_page,
        )

        # 直接放置 LiveCaptionPage（不使用滾動區），以便 resizeEvent 在實際小部件上生效
        self.tabs.addTab(self.live_caption_page, "即時字幕")
        self.tabs.addTab(self.settings_page, "設定")

        # 將 清空字幕與 開始 統一放在 tab bar 同一高度（右側）
        self._tab_corner_widget = QWidget()
        tab_corner_layout = QHBoxLayout(self._tab_corner_widget)
        tab_corner_layout.setContentsMargins(0, 0, 0, 0)
        tab_corner_layout.setSpacing(6)
        self._clear_caption_btn = QPushButton("清空字幕")
        self._clear_caption_btn.clicked.connect(self.clear_live_caption)
        self._start_session_btn = QPushButton("開始")
        self._start_session_btn.clicked.connect(self.start_session)

        # 調整角落按鈕大小與即時字幕/設定分頁按鈕一致
        tab_height = self.tabs.tabBar().sizeHint().height() or 40
        for btn in (self._clear_caption_btn, self._start_session_btn):
            btn.setFixedHeight(tab_height)
            btn.setMinimumWidth(96)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        tab_corner_layout.addWidget(self._clear_caption_btn)
        tab_corner_layout.addWidget(self._start_session_btn)
        self.tabs.setCornerWidget(self._tab_corner_widget, Qt.TopRightCorner)

        self._config_apply_service = ConfigApplyService(
            meeting_capture=self.meeting_capture,
            local_capture=self.local_capture,
            speaker_playback=self.speaker_playback,
            meeting_playback=self.meeting_playback,
            audio_routing_page=self.audio_routing_page,
            live_caption_page=self.live_caption_page,
            local_ai_page=self.local_ai_page,
            device_volume_controller=device_volume_controller,
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
        self._live_caption_apply_timer = QTimer(self)
        self._live_caption_apply_timer.setSingleShot(True)
        self._live_caption_apply_timer.setInterval(120)
        self._live_caption_apply_timer.timeout.connect(self._apply_live_caption_config_now)

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
        self.config = self._settings_service.load()
        self._runtime_facade.mark_dirty()
        self._ensure_pipelines_ready()
        self.refresh_from_system()
        self.statusBar().showMessage(f"已重載設定: {self.config_path}")

    def validate_current_routes(self) -> None:
        is_running = self.session_controller.is_running() if self.session_controller else False
        self.live_caption_page.set_start_enabled(True)
        label = "停止" if is_running else "開始"
        self.live_caption_page.set_start_label(label)
        self._start_session_btn.setText(label)
        self.live_caption_page.set_direction_controls_enabled(not self._session_action_running)
        route_message, route_has_error = self._current_route_validation_message()
        self.audio_routing_page.set_validation_message(route_message, is_error=route_has_error)
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
        self._schedule_live_caption_apply()

    def _schedule_live_apply(self) -> None:
        if not self._live_apply_ready or self._suspend_live_apply:
            return
        if self._session_action_running:
            self._pending_live_apply = True
            self.statusBar().showMessage("設定已變更，待開始或停止完成後套用")
            return
        if self.session_controller and self.session_controller.is_running():
            self._pending_live_apply = True
            self.statusBar().showMessage("設定已變更，將在停止 session 後套用")
            return
        self._live_apply_timer.start()

    def _schedule_live_caption_apply(self) -> None:
        if not self._live_apply_ready or self._suspend_live_apply:
            return
        if self._session_action_running:
            self._pending_live_caption_apply = True
            self.statusBar().showMessage("即時字幕設定已變更，待開始或停止完成後立即套用")
            return
        if self.session_controller and self.session_controller.is_running():
            self._live_caption_apply_timer.start()
            return
        self._schedule_live_apply()

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
            self.statusBar().showMessage("設定已變更，將在停止 session 後套用")
            return
        try:
            self._pending_live_apply = False
            self._sync_ui_to_config()
            self.live_caption_page.apply_config(self.config)
            self._runtime_facade.mark_dirty()
            path = Path(self.config_path)
            self._ensure_pipelines_ready()
            self.statusBar().showMessage(f"設定已更新: {path}")
            self.validate_current_routes()
        except Exception as exc:
            self._report_error(f"auto_apply_config failed: {exc}")
            self.statusBar().showMessage(f"設定套用失敗: {exc}")

    def _apply_live_caption_config_now(self) -> None:
        if not self._live_apply_ready:
            return
        if self._session_action_running:
            self._pending_live_caption_apply = True
            return
        if self._live_caption_apply_timer.isActive():
            self._live_caption_apply_timer.stop()

        is_running = self.session_controller.is_running() if self.session_controller else False
        if not is_running:
            self._apply_live_config_now()
            return

        try:
            self._pending_live_caption_apply = False
            self._config_apply_service.sync_live_caption_to_config(self.config)
            self.live_caption_page.apply_config(self.config)
            if self.audio_router:
                self.audio_router.refresh_runtime_config(self.config)
            self._apply_output_switches_to_router()
            self.validate_current_routes()
            self.statusBar().showMessage("即時字幕設定已熱切換")
        except Exception as exc:
            self._report_error(f"live_caption_hot_apply failed: {exc}")
            self.statusBar().showMessage(f"即時字幕熱切換失敗: {exc}")

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
        try:
            self._validate_route_devices_or_raise(route)
            self._sync_ui_to_config()
            self._runtime_facade.mark_dirty()
            self._ensure_pipelines_ready()
            self.clear_live_caption()
            self._run_session_action(
                "start",
                route=route,
                sample_rate=self.config.runtime.sample_rate,
                chunk_ms=self.config.runtime.chunk_ms,
                mode=self.live_caption_page.selected_mode(),
            )
        except Exception as exc:
            self._report_error(f"start_session failed: {exc}")
            QMessageBox.critical(self, "啟動失敗", str(exc))

    def _validate_route_devices_or_raise(self, route) -> None:
        message, has_error = self._describe_route_validation(route)
        if has_error:
            raise ValueError(message)

    def _current_route_validation_message(self) -> tuple[str, bool]:
        route = self.audio_routing_page.selected_audio_routes()
        return self._describe_route_validation(route)

    def _describe_route_validation(self, route) -> tuple[str, bool]:
        missing: list[str] = []
        unavailable: list[str] = []
        checks = (
            ("遠端輸入", route.meeting_in, self.input_device_names),
            ("本地輸入", route.microphone_in, self.input_device_names),
            ("本地輸出", route.speaker_out, self.output_device_names),
            ("遠端輸出", route.meeting_out, self.output_device_names),
        )
        for label, selector, existing_names in checks:
            raw_selector = str(selector or "").strip()
            if not raw_selector:
                missing.append(label)
                continue
            resolved = canonical_device_name(raw_selector).strip()
            if resolved and resolved not in existing_names:
                unavailable.append(f"{label}: {resolved}")
        if not missing and not unavailable:
            return "音訊裝置已就緒，可以開始即時翻譯。", False
        details: list[str] = []
        if missing:
            details.append("未設定裝置：" + "、".join(missing))
        if unavailable:
            details.append("找不到裝置：" + "；".join(unavailable))
        return "；".join(details), True

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
        if self.windowHandle() and not self.windowIcon().isNull():
            try:
                self.windowHandle().setIcon(self.windowIcon())
            except Exception:
                pass
        if self._window_geometry_ready:
            return
        self._window_geometry_ready = True
        self._ensure_window_decorations()
        QTimer.singleShot(0, self._fit_window_to_screen)

    def _set_initial_window_geometry(self) -> None:
        default_width = self._preferred_client_width()
        default_height = 1060
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

    def _preferred_client_width(self) -> int:
        page_width = 0
        if hasattr(self, "settings_page") and self.settings_page is not None:
            try:
                page_width = max(
                    self.settings_page.sizeHint().width(),
                    self.settings_page.minimumSizeHint().width(),
                )
            except Exception:
                page_width = self.settings_page.minimumWidth()
        return max(1280, min(1520, page_width + 80))

    def _preferred_client_height(self) -> int:
        page_height = 0
        if hasattr(self, "settings_page") and self.settings_page is not None:
            try:
                page_height = max(
                    self.settings_page.sizeHint().height(),
                    self.settings_page.minimumSizeHint().height(),
                )
            except Exception:
                page_height = self.settings_page.minimumHeight()
        tab_bar_h = self.tabs.tabBar().sizeHint().height() if hasattr(self, "tabs") else 0
        status_h = self.statusBar().sizeHint().height() if self.statusBar() else 0
        return max(820, min(1160, page_height + tab_bar_h + status_h + 28))

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
        for candidate in self._icon_candidates():
            icon = QIcon(str(candidate))
            if not icon.isNull():
                self._resolved_window_icon = icon
                self.setWindowIcon(icon)
                return

    @staticmethod
    def _icon_candidates() -> list[Path]:
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "lioil.ico")
            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                candidates.append(Path(meipass) / "lioil.ico")
        candidates.append(Path.cwd() / "lioil.ico")
        candidates.append(Path(__file__).resolve().parent.parent.parent / "lioil.ico")
        deduped: list[Path] = []
        seen: set[str] = set()
        for item in candidates:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _wrap_in_scroll_area(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
        # 降低最小高度以避免不必要的窗口溢出，並讓四個區塊在較小窗體中自動等比縮放
        desired_height = max(720, self._preferred_client_height() - 12)
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
        self._refresh_runtime_diagnostics()
        self._apply_detected_asr_labels()
        remote_original_items = self.transcript_buffer.latest("meeting_original", limit=2000)
        remote_translated_items = self.transcript_buffer.latest("meeting_translated", limit=2000)
        local_original_items = self.transcript_buffer.latest("local_original", limit=2000)
        local_translated_items = self.transcript_buffer.latest("local_translated", limit=2000)
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

    def _refresh_runtime_diagnostics(self) -> None:
        if not getattr(self, "diagnostics_page", None):
            return
        if not self.audio_router:
            self.diagnostics_page.set_asr_runtime_details("router=not-built")
            self.diagnostics_page.set_llm_runtime_details("")
            self.diagnostics_page.set_tts_runtime_details("")
            return
        try:
            router_stats = self.audio_router.stats()
        except Exception as exc:
            self.diagnostics_page.set_asr_runtime_details(f"stats-error={exc}")
            return
        self.diagnostics_page.set_asr_runtime_details(self._build_asr_diagnostics_summary(router_stats))
        self.diagnostics_page.set_llm_runtime_details(self._build_llm_diagnostics_summary(router_stats))
        self.diagnostics_page.set_tts_runtime_details(self._build_tts_diagnostics_summary(router_stats))

    @staticmethod
    def _build_asr_diagnostics_summary(router_stats) -> str:
        parts: list[str] = []
        for label, source in (("meeting", "remote"), ("local", "local")):
            asr = router_stats.asr.get(source) or {}
            parts.append(f"{label}:{MainWindow._build_asr_observation_fragment(asr)}")
        return " ; ".join(parts)

    @staticmethod
    def _build_asr_observation_fragment(asr: dict[str, object]) -> str:
        post = asr.get("postprocessor") or {}
        final_post = post.get("final") or {}
        rejected = int(final_post.get("rejected_count", 0) or 0)
        last_reason = str(final_post.get("last_rejection_reason", "") or "")
        queue = int(asr.get("queue_size", 0) or 0)
        partials = int(asr.get("partial_count", 0) or 0)
        finals = int(asr.get("final_count", 0) or 0)
        degradation = str(asr.get("degradation_level", "") or "").strip() or "-"
        signal = asr.get("endpoint_signal") or {}
        pause_ms = int(signal.get("pause_ms", 0) or 0)
        endpointing = asr.get("endpointing") or {}
        soft_count = int(endpointing.get("soft_endpoint_count", 0) or 0)
        hard_count = int(endpointing.get("hard_endpoint_count", 0) or 0)
        speech_started = int(endpointing.get("speech_started_count", 0) or 0)
        fragment = (
            f"q={queue} p={partials} f={finals} "
            f"pause={pause_ms} ep={speech_started}/{soft_count}/{hard_count} "
            f"deg={degradation} rej={rejected}"
        )
        if last_reason:
            fragment += f"({last_reason})"
        return fragment

    @staticmethod
    def _build_llm_diagnostics_summary(router_stats) -> str:
        overflow = router_stats.translation_overflow or {}
        latest_latency = (router_stats.latency or [{}])[0] if router_stats.latency else {}
        overflow_text = f"overflow l={int(overflow.get('local', 0))} r={int(overflow.get('remote', 0))}"
        if not latest_latency:
            return overflow_text
        return (
            f"{overflow_text} ; latest="
            f"{latest_latency.get('source', '-')}:"
            f"asr_final={latest_latency.get('speech_end_to_asr_final_ms', '-')}"
            f"/llm_final={latest_latency.get('asr_final_to_llm_final_ms', '-')}"
        )

    @staticmethod
    def _build_tts_diagnostics_summary(router_stats) -> str:
        tts = router_stats.tts or {}
        return (
            f"depth={int(tts.get('queue_depth', 0) or 0)} "
            f"local={int(tts.get('queue_depth_local', 0) or 0)} "
            f"remote={int(tts.get('queue_depth_remote', 0) or 0)} "
            f"drop_l={int(tts.get('drop_count_local', 0) or 0)} "
            f"drop_r={int(tts.get('drop_count_remote', 0) or 0)}"
        )

    def _resolve_active_sources(self) -> set[str]:
        if self._session_action_running and self._session_action_name == "start":
            return {"local", "remote"}

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

    def _on_output_mode_changed(self, mode: str) -> None:
        self._apply_output_switches_to_router()

    @staticmethod
    def _format_transcript_line(text: str, is_final: bool, speaker_label: str = "") -> str:
        if _S2T_CONVERTER is not None:
            try:
                text = _S2T_CONVERTER.convert(text)
            except Exception:
                pass
        state = "final" if is_final else "partial"
        speaker_prefix = f"{speaker_label}: " if speaker_label else ""
        return f"[{state}] {speaker_prefix}{text}"

    @classmethod
    def _build_transcript_lines(cls, items) -> list[str]:
        # newest first: 讓最新字幕顯示在畫面最上方
        return [
            cls._format_transcript_line(item.text, item.is_final, getattr(item, "speaker_label", ""))
            for item in reversed(items)
        ]

    def _report_error(self, message: str | ErrorEvent) -> None:
        text = message.to_log_line() if isinstance(message, ErrorEvent) else str(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = runtime_logs_dir() / "runtime_events.log"
        with self._error_lock:
            line = self._record_error_line_locked(text=text, timestamp=timestamp)
            if not line:
                return
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    def export_diagnostics(self) -> None:
        routes = self.audio_routing_page.selected_audio_routes()
        output_path = self._export_service.export_runtime(
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
            on_asr_event=self._on_asr_stage_event,
            on_diagnostic_event=lambda message: self._report_error(f"[router] {message}"),
        )

    def _on_asr_stage_event(self, event: object) -> None:
        source = str(getattr(event, "source", "") or "")
        if source not in {"local", "remote"}:
            return
        detected = str(getattr(event, "detected_language", "") or "").strip().lower()
        if not detected:
            return
        with self._asr_detected_lock:
            self._latest_asr_detected_lang[source] = detected

    def _apply_detected_asr_labels(self) -> None:
        with self._asr_detected_lock:
            snapshot = dict(self._latest_asr_detected_lang)
        for source, language in snapshot.items():
            if language:
                self.live_caption_page.set_detected_asr_language(source, language)
        self.live_caption_page.update_translation_voice_labels(self.config)

    def _build_pipelines_from_config(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.session_controller.stop()
        bundle = self._runtime_facade.rebuild(self.config)
        self.audio_router = bundle.audio_router
        self.session_controller = bundle.session_controller
        self._apply_output_switches_to_router()

    def _ensure_pipelines_ready(self) -> None:
        bundle = self._runtime_facade.ensure_ready(self.config)
        self.audio_router = bundle.audio_router
        self.session_controller = bundle.session_controller
        self._apply_output_switches_to_router()

    def _apply_output_switches_to_router(self) -> None:
        if not self.audio_router:
            return
        local_mode = self.live_caption_page.selected_tts_output_mode_for_channel("local")
        remote_mode = self.live_caption_page.selected_tts_output_mode_for_channel("remote")
        # UI controls are organized by transcript/source panel, while router output modes
        # are organized by playback destination channel. Remote-source output plays locally,
        # and local-source output plays remotely, so the mapping is intentionally crossed.
        self.audio_router.set_output_mode("local", remote_mode)
        self.audio_router.set_output_mode("remote", local_mode)

    def run_system_check(self) -> None:
        if self._healthcheck_service.running:
            self.statusBar().showMessage("健康檢查進行中")
            return
        self._sync_ui_to_config()
        try:
            started = self._healthcheck_service.start(config=self.config)
        except Exception as exc:
            self._report_error(f"system_check start failed: {exc}")
            self.statusBar().showMessage("健康檢查啟動失敗")
            return
        if not started:
            self.statusBar().showMessage("健康檢查進行中")
            return
        self.local_ai_page.set_runtime_status("健康檢查進行中")
        self.diagnostics_page.set_health_details("健康檢查進行中...")
        self.statusBar().showMessage("健康檢查進行中...")

    def _drain_health_check_results(self) -> None:
        update = self._healthcheck_service.poll()
        if update is None:
            return
        if update.kind == "timeout":
            summary = "健康檢查：逾時"
            message = update.message
            self.local_ai_page.set_runtime_status(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(message)
            self._report_error(message)
            return

        if update.kind == "error":
            message = update.message
            summary = "健康檢查：失敗"
            self.local_ai_page.set_runtime_status(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(summary)
            self._report_error(f"system_check failed: {message}")
            return

        report = update.report
        if report is None:
            self._report_error("system_check failed: report is missing")
            self.statusBar().showMessage("健康檢查：失敗")
            return
        summary = "健康檢查：正常" if report.ok else "健康檢查：有異常"
        self.local_ai_page.set_runtime_status(summary)
        self.diagnostics_page.set_asr_details(f"{'正常' if report.asr_ok else '異常'} - {report.asr_message}")
        self.diagnostics_page.set_llm_details(f"{'正常' if report.llm_ok else '異常'} - {report.llm_message}")
        self.diagnostics_page.set_tts_details(f"{'正常' if report.tts_ok else '異常'} - {report.tts_message}")
        self.statusBar().showMessage(summary)

    def _run_session_action(
        self,
        action: str,
        *,
        route=None,
        sample_rate: int | None = None,
        chunk_ms: int | None = None,
        mode: str | None = None,
    ) -> None:
        if not self.session_controller or self._session_action_running:
            return
        self._session_action_running = True
        self._session_action_name = action
        self.live_caption_page.set_start_enabled(False)
        if action == "stop":
            self.live_caption_page.set_start_label("停止中...")
            self._start_session_btn.setText("停止中...")
            self.statusBar().showMessage("正在停止 session...")
        else:
            self.live_caption_page.set_start_label("啟動中...")
            self._start_session_btn.setText("啟動中...")
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
                    if route is None or sample_rate is None:
                        raise ValueError("missing session start parameters")
                    result = self.session_controller.start(
                        route,
                        sample_rate=sample_rate,
                        chunk_ms=chunk_ms or self.config.runtime.chunk_ms,
                        mode=str(mode or getattr(self.config.direction, "mode", "bidirectional") or "bidirectional"),
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
        if self._pending_live_caption_apply:
            QTimer.singleShot(0, self._apply_live_caption_config_now)
        if action == "stop" and self._pending_live_apply:
            QTimer.singleShot(0, self._apply_live_config_now)

    def _export_session_report(self, payload: dict[str, object]) -> None:
        try:
            routes = self.audio_routing_page.selected_audio_routes()
            self._export_service.export_session(
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
            self._flush_error_summaries_locked()
            return list(self.recent_errors)

    def _record_error_line_locked(self, *, text: str, timestamp: str) -> str:
        now = time.monotonic()
        state = self._error_throttle_state.setdefault(
            text,
            {"last_emit_ts": 0.0, "suppressed_count": 0},
        )
        last_emit_ts = float(state.get("last_emit_ts", 0.0))
        if last_emit_ts > 0.0 and (now - last_emit_ts) <= self._error_throttle_window_sec:
            state["suppressed_count"] = int(state.get("suppressed_count", 0)) + 1
            return ""
        self._flush_single_error_summary_locked(text=text, timestamp=timestamp)
        line = f"{timestamp} {text}"
        self.recent_errors.append(line)
        state["last_emit_ts"] = now
        return line

    def _flush_error_summaries_locked(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for text in list(self._error_throttle_state.keys()):
            self._flush_single_error_summary_locked(text=text, timestamp=timestamp)

    def _flush_single_error_summary_locked(self, *, text: str, timestamp: str) -> None:
        state = self._error_throttle_state.get(text)
        if not state:
            return
        suppressed = int(state.get("suppressed_count", 0))
        if suppressed <= 0:
            return
        self.recent_errors.append(f"{timestamp} [dedup] '{text}' repeated {suppressed} more times")
        state["suppressed_count"] = 0

    def _build_runtime_stats_text(self) -> str:
        lines = [
            f"remote_translation_target: {self.config.language.meeting_target}",
            f"local_translation_target: {self.config.language.local_target}",
            f"remote_translation_enabled: {translation_enabled_for_source(self.config.runtime, 'remote')}",
            f"local_translation_enabled: {translation_enabled_for_source(self.config.runtime, 'local')}",
            "asr_language_mode: auto",
            f"tts_output_mode: {str(getattr(self.config.runtime, 'tts_output_mode', 'subtitle_only') or 'subtitle_only')}",
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
                    f"vad_rms={asr['vad_rms']:.5f} vad_th={asr['vad_threshold']:.5f} "
                    f"adaptive={asr.get('adaptive_mode', '-')} "
                    f"partial_ms={asr.get('adaptive_partial_interval_ms', '-')} "
                    f"silence_ms={asr.get('adaptive_min_silence_duration_ms', '-')} "
                    f"soft_final_ms={asr.get('adaptive_soft_final_audio_ms', '-')} "
                    f"backend={asr.get('resolved_backend', '-')} "
                    f"lang_family={asr.get('resolved_language_family', '-')} "
                    f"device={asr.get('device_effective', '-')} "
                    f"init={asr.get('model_init_mode', '-')}"
                )
            )
            if capture["last_error"]:
                lines.append(f"{label}: capture_error={capture['last_error']}")
            if asr["last_debug"]:
                lines.append(f"{label}: asr_last={asr['last_debug']}")
            final_post = ((asr.get("postprocessor") or {}).get("final") or {})
            partial_post = ((asr.get("postprocessor") or {}).get("partial") or {})
            if final_post:
                lines.append(
                    f"{label}: post_final accepted={final_post.get('accepted_count', 0)} "
                    f"rejected={final_post.get('rejected_count', 0)} "
                    f"last_reason={final_post.get('last_rejection_reason', '-') or '-'}"
                )
            if partial_post:
                lines.append(
                    f"{label}: post_partial accepted={partial_post.get('accepted_count', 0)} "
                    f"rejected={partial_post.get('rejected_count', 0)} "
                    f"last_reason={partial_post.get('last_rejection_reason', '-') or '-'}"
                )
            if asr.get("backend_resolution_reason"):
                lines.append(f"{label}: backend_reason={asr['backend_resolution_reason']}")
            if asr.get("init_failure"):
                lines.append(f"{label}: init_failure={asr['init_failure']}")
        tts = router_stats.tts
        lines.append(
            "tts: "
            f"depth={tts['queue_depth']} local_depth={tts['queue_depth_local']} remote_depth={tts['queue_depth_remote']} "
            f"drop_local={tts['drop_count_local']} drop_remote={tts['drop_count_remote']} "
            f"oldest_age_ms={tts['oldest_age_ms']:.1f}"
        )
        if router_stats.latency:
            latest = router_stats.latency[0]
            lines.append(
                "latency: "
                f"source={latest.get('source', '-')} "
                f"first_asr_partial_ms={latest.get('first_asr_partial_ms', '-')} "
                f"first_display_partial_ms={latest.get('first_display_partial_ms', '-')} "
                f"speech_end_to_asr_final_ms={latest.get('speech_end_to_asr_final_ms', '-')} "
                f"asr_final_to_llm_final_ms={latest.get('asr_final_to_llm_final_ms', '-')} "
                f"tts_enqueue_to_playback_start_ms={latest.get('tts_enqueue_to_playback_start_ms', '-')} "
                f"tts_enqueue_kind={latest.get('tts_enqueue_kind', '-')}"
            )
        return "\n".join(lines)

    def _sync_ui_to_config(self) -> None:
        self._config_apply_service.sync_ui_to_config(self.config)

    def _apply_audio_route_levels(self) -> None:
        self._config_apply_service.apply_audio_route_levels(self.config.audio)

    def _apply_audio_route_levels_from_ui(self) -> None:
        self._config_apply_service.apply_audio_route_levels_from_ui(self.config)

    def _save_config_to_disk(self) -> Path:
        return self._settings_service.save(self.config)



