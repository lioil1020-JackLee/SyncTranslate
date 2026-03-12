from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock
from threading import Thread
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QIcon, QShowEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget
import numpy as np

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.debug_panel import DebugPanel
from app.device_manager import DeviceManager
from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.streaming_asr import StreamingAsr
from app.local_ai.tts_factory import create_tts_engine
from app.local_ai.translation_stitcher import TranslationStitcher
from app.local_ai.vad_segmenter import VadConfig, VadSegmenter
from app.pages_audio_routing import AudioRoutingPage
from app.pages_diagnostics import DiagnosticsPage
from app.pages_io_control import IoControlPage
from app.pages_live_caption import LiveCaptionPage
from app.pages_local_ai import LocalAiPage
from app.pipeline_direction import DirectionalPipeline
from app.router import check_routes
from app.schemas import AppConfig, TtsConfig
from app.session_controller import SessionController
from app.settings import load_config, save_config
from app.transcript_buffer import TranscriptBuffer


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
        self.meeting_pipeline: DirectionalPipeline | None = None
        self.local_pipeline: DirectionalPipeline | None = None
        self.session_controller: SessionController | None = None
        self._health_check_running = False
        self._health_check_queue: Queue[tuple[bool, object]] = Queue()
        self._health_check_started_at = 0.0
        self._health_check_timeout_sec = 45.0
        self._session_action_running = False
        self._session_action_queue: Queue[tuple[str, bool, object]] = Queue()
        self._pending_live_apply = False
        self._pipelines_dirty = False
        self._volume_sync_queue: Queue[dict[str, float | None]] = Queue(maxsize=1)
        self._volume_sync_running = False

        self.setWindowTitle("SyncTranslate - Local AI Runtime")
        self._set_window_icon()

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.audio_routing_page = AudioRoutingPage(
            on_apply_banana_preset=self.apply_banana_preset,
            on_save=self.persist_config,
            on_reload=self.reload_config,
            on_route_changed=self._on_audio_routing_changed,
        )
        self.live_caption_page = LiveCaptionPage(
            on_clear_clicked=self.clear_live_caption,
            on_start_clicked=self.start_session,
            on_settings_changed=self._on_live_caption_settings_changed,
        )
        self.local_ai_page = LocalAiPage(
            on_settings_changed=self._on_local_ai_changed,
            on_health_check=self.run_health_check,
        )
        self.diagnostics_page = DiagnosticsPage(
            on_health_check=self.run_health_check,
            on_test_meeting_tts=self.test_meeting_tts_output,
            on_test_speaker_tts=self.test_speaker_tts_output,
            on_export_diagnostics=self.export_diagnostics,
        )
        self.io_control_page = IoControlPage(self.audio_routing_page, self.diagnostics_page)
        self.debug_panel = DebugPanel()

        self.tabs.addTab(self._wrap_in_scroll_area(self.io_control_page), "音訊路由與診斷")
        self.tabs.addTab(self._wrap_in_scroll_area(self.live_caption_page), "即時字幕")
        self.tabs.addTab(self.local_ai_page, "參數設定")
        self.tabs.addTab(self._wrap_in_scroll_area(self.debug_panel), "除錯")
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
        self.volume_sync_timer = QTimer(self)
        self.volume_sync_timer.setInterval(1500)
        self.volume_sync_timer.timeout.connect(self._sync_system_volume_sliders)
        self.volume_sync_timer.start()
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
        if self.config.runtime.warmup_on_start:
            QTimer.singleShot(100, lambda: self.run_health_check(True))

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

    def _sync_system_volume_sliders(self) -> None:
        self._drain_volume_sync_results()
        if self._volume_sync_running:
            return
        routes = self.audio_routing_page.selected_audio_routes()
        self._volume_sync_running = True

        def _worker() -> None:
            result: dict[str, float | None]
            try:
                from app.windows_volume import get_input_volume, get_output_volume

                result = {
                    "meeting_in": None if self.audio_routing_page._should_skip_system_sync(routes.meeting_in) else get_input_volume(routes.meeting_in),
                    "microphone_in": None if self.audio_routing_page._should_skip_system_sync(routes.microphone_in) else get_input_volume(routes.microphone_in),
                    "speaker_out": None if self.audio_routing_page._should_skip_system_sync(routes.speaker_out) else get_output_volume(routes.speaker_out),
                    "meeting_out": None if self.audio_routing_page._should_skip_system_sync(routes.meeting_out) else get_output_volume(routes.meeting_out),
                }
            except Exception as exc:
                self._report_error(f"volume_sync failed: {exc}")
                result = {
                    "meeting_in": None,
                    "microphone_in": None,
                    "speaker_out": None,
                    "meeting_out": None,
                }
            try:
                while not self._volume_sync_queue.empty():
                    self._volume_sync_queue.get_nowait()
            except Empty:
                pass
            self._volume_sync_queue.put(result)

        Thread(target=_worker, daemon=True).start()

    def _drain_volume_sync_results(self) -> None:
        try:
            result = self._volume_sync_queue.get_nowait()
        except Empty:
            return
        self._volume_sync_running = False
        if self.audio_routing_page.sync_system_volume_sliders(result):
            self._apply_audio_route_levels_from_ui()

    def apply_banana_preset(self) -> None:
        self.config.audio.meeting_in = "Windows DirectSound::Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)"
        self.config.audio.meeting_out = "Windows DirectSound::Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)"
        self.audio_routing_page.apply_config(self.config)
        self.audio_routing_page.set_status("已套用 Banana 預設（請確認麥克風輸入 / 喇叭輸出）")
        self.validate_current_routes()
        self._schedule_live_apply()

    def persist_config(self) -> None:
        self._apply_live_config_now()
        self.audio_routing_page.set_status(f"設定已儲存: {self.config_path}")

    def reload_config(self) -> None:
        if self.session_controller:
            self.session_controller.stop()
        self.config = load_config(self.config_path)
        self._pipelines_dirty = True
        self._ensure_pipelines_ready()
        self.refresh_from_system()
        self.audio_routing_page.set_status(f"已重載設定: {self.config_path}")

    def validate_current_routes(self) -> None:
        current_routes = self.audio_routing_page.selected_audio_routes()
        current_mode = self.audio_routing_page.selected_mode()
        result = check_routes(
            current_routes,
            self.input_device_names,
            self.output_device_names,
            mode=current_mode,
        )
        self.debug_panel.update_route_result(result)
        self.debug_panel.update_recent_errors(self._get_recent_errors())
        is_running = self.session_controller.is_running() if self.session_controller else False
        self.live_caption_page.set_start_enabled(result.ok or is_running)
        self.live_caption_page.set_start_label("停止" if is_running else "開始")
        if result.ok:
            self.audio_routing_page.set_status("路由檢查: OK")
        else:
            self.audio_routing_page.set_status("路由檢查: Error，請到除錯頁查看")
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
        mode = self.audio_routing_page.selected_mode()
        try:
            self._pending_live_apply = False
            self._sync_ui_to_config()
            self._pipelines_dirty = True
            path = self._save_config_to_disk()

            if was_running and self.session_controller:
                result = self.session_controller.start(
                    mode,
                    route,
                    sample_rate=self.config.runtime.sample_rate,
                    chunk_ms=self.config.runtime.chunk_ms,
                )
                if not result.ok:
                    raise ValueError(result.message)
                self.statusBar().showMessage(f"設定已套用並重新啟動: {path}")
            else:
                self.statusBar().showMessage(f"設定已套用: {path}")
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
        mode = self.audio_routing_page.selected_mode()
        try:
            self._sync_ui_to_config()
            self._pipelines_dirty = True
            self._ensure_pipelines_ready()
            self._save_config_to_disk()
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
        self.transcript_buffer.clear()
        self._remote_line_cache = []
        self._remote_translated_line_cache = []
        self._local_line_cache = []
        self._local_translated_line_cache = []
        self.statusBar().showMessage("字幕已清空")

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._sync_ui_to_config()
            self._save_config_to_disk()
        except Exception as exc:
            self._report_error(f"save_on_close failed: {exc}")
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

    def _get_speaker_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().speaker_out

    def _get_meeting_output_device(self) -> str:
        return self.audio_routing_page.selected_audio_routes().meeting_out

    def test_meeting_tts_output(self) -> None:
        device_name = self._get_meeting_output_device()
        try:
            self._sync_ui_to_config()
            audio, sample_rate, engine_label = self._build_output_test_audio(
                primary_tts=self.config.local_tts,
                text=self._tts_test_text(self.config.language.local_target),
            )
            self.meeting_playback.play(
                audio=audio,
                sample_rate=sample_rate,
                output_device_name=device_name,
                blocking=True,
            )
            self.statusBar().showMessage(f"遠端輸出測試完成: {device_name} ({engine_label})")
        except Exception as exc:
            self._report_error(f"test_meeting_tts_output failed: {exc}")
            QMessageBox.critical(self, "測試失敗", str(exc))

    def test_speaker_tts_output(self) -> None:
        device_name = self._get_speaker_output_device()
        try:
            self._sync_ui_to_config()
            audio, sample_rate, engine_label = self._build_output_test_audio(
                primary_tts=self.config.meeting_tts,
                text=self._tts_test_text(self.config.language.meeting_target),
            )
            self.speaker_playback.play(
                audio=audio,
                sample_rate=sample_rate,
                output_device_name=device_name,
                blocking=True,
            )
            self.statusBar().showMessage(f"本地喇叭測試完成: {device_name} ({engine_label})")
        except Exception as exc:
            self._report_error(f"test_speaker_tts_output failed: {exc}")
            QMessageBox.critical(self, "測試失敗", self._format_playback_error(exc))

    @staticmethod
    def _format_transcript_line(text: str, is_final: bool) -> str:
        state = "final" if is_final else "partial"
        return f"[{state}] {text}"

    @classmethod
    def _build_transcript_lines(cls, items) -> list[str]:
        return [cls._format_transcript_line(item.text, item.is_final) for item in items]

    def _report_error(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path("logs") / "runtime_events.log"
        log_path.parent.mkdir(exist_ok=True)
        with self._error_lock:
            line = f"{timestamp} {message}"
            self.recent_errors.append(line)
        try:
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    def export_diagnostics(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"diagnostics_{now}.txt")
        routes = self.audio_routing_page.selected_audio_routes()
        content = "\n".join(
            [
                "SyncTranslate Diagnostics",
                f"timestamp: {datetime.now().isoformat()}",
                f"config_path: {self.config_path}",
                f"mode: {self.config.direction.mode}",
                f"meeting_language: {self.config.language.meeting_source} -> {self.config.language.meeting_target}",
                f"local_language: {self.config.language.local_source} -> {self.config.language.local_target}",
                f"meeting_in: {routes.meeting_in}",
                f"microphone_in: {routes.microphone_in}",
                f"speaker_out: {routes.speaker_out}",
                f"meeting_out: {routes.meeting_out}",
                f"asr_model: {self.config.asr.model}",
                f"asr_device: {self.config.asr.device}",
                f"asr_compute_type: {self.config.asr.compute_type}",
                f"llm_backend: {self.config.llm.backend}",
                f"llm_url: {self.config.llm.base_url}",
                f"llm_model: {self.config.llm.model}",
                f"meeting_tts_engine: {self.config.meeting_tts.engine}",
                f"meeting_tts_model: {self.config.meeting_tts.model_path}",
                f"meeting_tts_voice: {self.config.meeting_tts.voice_name}",
                f"local_tts_engine: {self.config.local_tts.engine}",
                f"local_tts_model: {self.config.local_tts.model_path}",
                f"local_tts_voice: {self.config.local_tts.voice_name}",
                f"sample_rate: {self.config.runtime.sample_rate}",
                "runtime_stats:",
                self._build_runtime_stats_text(),
                "recent_errors:",
                *self._get_recent_errors()[-30:],
            ]
        )
        output_path.write_text(content, encoding="utf-8")
        self.statusBar().showMessage(f"已匯出診斷資訊: {output_path}")

    def _build_pipelines_from_config(self) -> None:
        if self.session_controller and self.session_controller.is_running():
            self.session_controller.stop()

        self.meeting_pipeline = self._create_pipeline(
            source_lang=self.config.language.meeting_source,
            target_lang=self.config.language.meeting_target,
            source_channel="meeting_original",
            translated_channel="meeting_translated",
            capture=self.meeting_capture,
            playback=self.speaker_playback,
            get_output_device=self._get_speaker_output_device,
            tts_config=self.config.meeting_tts,
        )
        self.local_pipeline = self._create_pipeline(
            source_lang=self.config.language.local_source,
            target_lang=self.config.language.local_target,
            source_channel="local_original",
            translated_channel="local_translated",
            capture=self.local_capture,
            playback=self.meeting_playback,
            get_output_device=self._get_meeting_output_device,
            tts_config=self.config.local_tts,
        )
        self.session_controller = SessionController(self.meeting_pipeline, self.local_pipeline)
        self._pipelines_dirty = False

    def _ensure_pipelines_ready(self) -> None:
        if (
            self.session_controller is None
            or self.meeting_pipeline is None
            or self.local_pipeline is None
            or self._pipelines_dirty
        ):
            self._build_pipelines_from_config()

    def _create_pipeline(
        self,
        *,
        source_lang: str,
        target_lang: str,
        source_channel: str,
        translated_channel: str,
        capture: AudioCapture,
        playback: AudioPlayback,
        get_output_device,
        tts_config: TtsConfig,
    ) -> DirectionalPipeline:
        asr_engine = FasterWhisperEngine(
            model=self.config.asr.model,
            device=self.config.asr.device,
            compute_type=self.config.asr.compute_type,
            beam_size=self.config.asr.beam_size,
            condition_on_previous_text=self.config.asr.condition_on_previous_text,
            language=source_lang,
        )
        vad = VadSegmenter(
            VadConfig(
                enabled=self.config.asr.vad.enabled,
                min_speech_duration_ms=self.config.asr.vad.min_speech_duration_ms,
                min_silence_duration_ms=self.config.asr.vad.min_silence_duration_ms,
                max_speech_duration_s=self.config.asr.vad.max_speech_duration_s,
                speech_pad_ms=self.config.asr.vad.speech_pad_ms,
                rms_threshold=self.config.asr.vad.rms_threshold,
            )
        )
        stream = StreamingAsr(
            engine=asr_engine,
            vad=vad,
            partial_interval_ms=self.config.asr.streaming.partial_interval_ms,
            partial_history_seconds=self.config.asr.streaming.partial_history_seconds,
            final_history_seconds=self.config.asr.streaming.final_history_seconds,
            queue_maxsize=self.config.runtime.asr_queue_maxsize,
            on_debug=self._report_error,
        )
        llm = OllamaClient(
            backend=self.config.llm.backend,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            request_timeout_sec=self.config.llm.request_timeout_sec,
        )
        stitcher = TranslationStitcher(
            translator=llm,
            source_lang=source_lang,
            target_lang=target_lang,
            enabled=self.config.llm.sliding_window.enabled,
            trigger_tokens=self.config.llm.sliding_window.trigger_tokens,
            max_context_items=self.config.llm.sliding_window.max_context_items,
        )
        return DirectionalPipeline(
            audio_capture=capture,
            transcript_buffer=self.transcript_buffer,
            audio_playback=playback,
            asr_stream=stream,
            stitcher=stitcher,
            tts=self._create_tts_engine(tts_config),
            source_channel=source_channel,
            translated_channel=translated_channel,
            get_output_device=get_output_device,
            tts_sample_rate=tts_config.sample_rate,
            on_error=self._report_error,
        )

    @staticmethod
    def _create_tts_engine(tts_config: TtsConfig):
        return create_tts_engine(tts_config)

    def run_health_check(self, warmup: bool) -> None:
        if self._health_check_running:
            self.statusBar().showMessage("health check 進行中")
            return
        self._sync_ui_to_config()
        config_path = str(self._save_config_to_disk())
        self._health_check_running = True
        self._health_check_started_at = time.monotonic()
        self.local_ai_page.set_runtime_status("健康檢查進行中")
        self.diagnostics_page.set_health_summary("健康檢查：進行中")
        self.diagnostics_page.set_health_details("健康檢查進行中...")
        self.statusBar().showMessage("健康檢查進行中...")

        def _worker() -> None:
            try:
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

        Thread(target=_worker, daemon=True).start()

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
            self.diagnostics_page.set_health_summary(summary)
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
            self.diagnostics_page.set_health_summary(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(summary)
            self._report_error(f"health_check failed: {message}")
            return

        report = payload
        summary = "健康檢查：正常" if report.ok else "健康檢查：失敗"
        detail = "\n".join(
            [
                f"ASR：{'正常' if report.asr_ok else '失敗'} - {report.asr_message}",
                f"LLM：{'正常' if report.llm_ok else '失敗'} - {report.llm_message}",
                f"TTS：{'正常' if report.tts_ok else '失敗'} - {report.tts_message}",
            ]
        )
        self.local_ai_page.set_runtime_status(summary)
        self.diagnostics_page.set_health_summary(summary)
        self.diagnostics_page.set_health_details(detail)
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
        self.live_caption_page.set_start_enabled(False)
        if action == "stop":
            self.live_caption_page.set_start_label("停止中...")
            self.statusBar().showMessage("正在停止 session...")
        else:
            self.live_caption_page.set_start_label("啟動中...")
            self.statusBar().showMessage("正在啟動 session...")

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
        self.validate_current_routes()
        if action == "stop" and self._pending_live_apply:
            QTimer.singleShot(0, self._apply_live_config_now)

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
        for label, pipeline in (("meeting", self.meeting_pipeline), ("local", self.local_pipeline)):
            if not pipeline:
                lines.append(f"{label}: pipeline not built")
                continue
            stats = pipeline.stats()
            lines.append(
                (
                    f"{label}: running={stats['running']} capture={stats['capture_running']} "
                    f"rate={int(stats['capture_rate']) if stats['capture_rate'] else 0} "
                    f"frames={stats['capture_frames']} level={stats['capture_level']:.5f} "
                    f"queue={stats['asr_queue']} dropped={stats['asr_dropped']} "
                    f"partials={stats['asr_partials']} finals={stats['asr_finals']} "
                    f"vad_rms={stats['vad_rms']:.5f} vad_th={stats['vad_threshold']:.5f}"
                )
            )
            if stats["capture_error"]:
                lines.append(f"{label}: capture_error={stats['capture_error']}")
            if stats["asr_last"]:
                lines.append(f"{label}: asr_last={stats['asr_last']}")
        return "\n".join(lines)

    def _sync_ui_to_config(self) -> None:
        self.config.audio = self.audio_routing_page.selected_audio_routes()
        self._apply_audio_route_levels()
        self.config.direction.mode = self.audio_routing_page.selected_mode()
        self.live_caption_page.update_config(self.config)
        self.local_ai_page.update_config(self.config)

    def _apply_audio_route_levels(self) -> None:
        self.meeting_capture.set_gain(self.config.audio.meeting_in_gain)
        self.local_capture.set_gain(self.config.audio.microphone_in_gain)
        self.speaker_playback.set_volume(self.config.audio.speaker_out_volume)
        self.meeting_playback.set_volume(self.config.audio.meeting_out_volume)

    def _apply_audio_route_levels_from_ui(self) -> None:
        audio = self.audio_routing_page.selected_audio_routes()
        self.meeting_capture.set_gain(audio.meeting_in_gain)
        self.local_capture.set_gain(audio.microphone_in_gain)
        self.speaker_playback.set_volume(audio.speaker_out_volume)
        self.meeting_playback.set_volume(audio.meeting_out_volume)

    def _save_config_to_disk(self) -> Path:
        return save_config(self.config, self.config_path)

    def _build_output_test_audio(self, *, primary_tts: TtsConfig, text: str) -> tuple[np.ndarray, int, str]:
        try:
            tts = self._create_tts_engine(primary_tts)
            audio = tts.synthesize(text, sample_rate=primary_tts.sample_rate)
            if audio.size == 0:
                raise ValueError("tts returned empty audio")
            return audio, primary_tts.sample_rate, "configured tts"
        except Exception as exc:
            engine = (primary_tts.engine or "").strip() or "unknown"
            voice = (primary_tts.voice_name or "").strip() or primary_tts.model_path
            raise ValueError(f"Configured TTS failed ({engine}, {voice}): {exc}") from exc

    def _candidate_test_tts_configs(self, primary_tts: TtsConfig) -> list[tuple[str, TtsConfig]]:
        candidates: list[tuple[str, TtsConfig]] = [("configured tts", primary_tts)]
        seen: set[tuple[str, str, str]] = {
            (primary_tts.engine, primary_tts.model_path, primary_tts.voice_name)
        }
        for label, config in (
            ("local piper", self.config.local_tts),
            ("meeting piper", self.config.meeting_tts),
            ("default piper", self._default_test_tts_config()),
        ):
            key = (config.engine, config.model_path, config.voice_name)
            if key in seen:
                continue
            if not self._is_local_test_tts_available(config):
                continue
            seen.add(key)
            candidates.append((label, config))
        return candidates

    @staticmethod
    def _is_local_test_tts_available(config: TtsConfig) -> bool:
        if (config.engine or "").strip().lower() != "piper":
            return False
        exe = Path(config.executable_path)
        model = Path(config.model_path)
        if not exe.is_absolute():
            exe = (Path.cwd() / exe).resolve()
        if not model.is_absolute():
            model = (Path.cwd() / model).resolve()
        return exe.exists() and model.exists()

    @staticmethod
    def _default_test_tts_config() -> TtsConfig:
        nested = Path(".\\tools\\piper\\piper\\piper.exe")
        flat = Path(".\\tools\\piper\\piper.exe")
        exe = str(nested if nested.exists() else flat)
        return TtsConfig(
            engine="piper",
            executable_path=exe,
            model_path=".\\models\\tts\\zh-TW-medium.onnx",
            config_path=".\\models\\tts\\zh-TW-medium.onnx.json",
            sample_rate=22050,
        )

    @staticmethod
    def _build_test_tone_audio() -> np.ndarray:
        sample_rate = 24000
        duration = 0.8
        t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        wave = 0.18 * np.sin(2 * np.pi * 523.25 * t)
        return wave.reshape(-1, 1).astype(np.float32)

    @staticmethod
    def _format_playback_error(exc: Exception) -> str:
        text = str(exc).strip()
        if "Unable to play TTS audio" in text:
            return "無法在選定的喇叭輸出播放測試音。請改選可用的喇叭或耳機裝置。\n\n詳細資訊:\n" + text
        return text

    @staticmethod
    def _tts_test_text(language: str) -> str:
        normalized = (language or "").lower()
        if normalized.startswith("zh"):
            return "這是翻譯語音測試。"
        if normalized.startswith("ja"):
            return "これは音声テストです。"
        return "This is a translated speech test."

