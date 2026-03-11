from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from threading import Thread
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication, QIcon, QShowEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QWidget
import numpy as np

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.debug_panel import DebugPanel
from app.device_manager import DeviceManager
from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.healthcheck import run_local_healthcheck
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

        self.setWindowTitle("SyncTranslate - Local AI Runtime")
        self._set_window_icon()
        self._set_initial_window_geometry()

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
        self.tabs.addTab(self.local_ai_page, "本地 AI")
        self.tabs.addTab(self._wrap_in_scroll_area(self.debug_panel), "除錯")
        self.setCentralWidget(self.tabs)
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
        finally:
            self._suspend_live_apply = False

        self.statusBar().showMessage(
            f"Config: {self.config_path} | input={len(input_devices)} output={len(output_devices)}"
        )
        self.validate_current_routes()

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
        self._build_pipelines_from_config()
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
        self._schedule_live_apply()

    def _on_local_ai_changed(self) -> None:
        self._schedule_live_apply()

    def _on_live_caption_settings_changed(self) -> None:
        self._schedule_live_apply()

    def _schedule_live_apply(self) -> None:
        if not self._live_apply_ready or self._suspend_live_apply:
            return
        self._live_apply_timer.start()

    def _apply_live_config_now(self) -> None:
        if not self._live_apply_ready:
            return
        if self._live_apply_timer.isActive():
            self._live_apply_timer.stop()

        was_running = self.session_controller.is_running() if self.session_controller else False
        route = self.audio_routing_page.selected_audio_routes()
        mode = self.audio_routing_page.selected_mode()
        try:
            self._sync_ui_to_config()
            self._build_pipelines_from_config()
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
        if not self.session_controller or self._session_action_running:
            return
        if self.session_controller.is_running():
            self._run_session_action("stop")
            return

        route = self.audio_routing_page.selected_audio_routes()
        mode = self.audio_routing_page.selected_mode()
        try:
            self._sync_ui_to_config()
            self._build_pipelines_from_config()
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
        QTimer.singleShot(0, self._fit_window_to_screen)

    def _set_initial_window_geometry(self) -> None:
        default_width = 1180
        default_height = 800
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
            audio, sample_rate, engine_label = self._build_output_test_audio(
                primary_tts=self.config.local_tts,
                text="會議輸出測試",
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
            audio, sample_rate, engine_label = self._build_output_test_audio(
                primary_tts=self.config.meeting_tts,
                text="喇叭測試",
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
            QMessageBox.critical(self, "測試失敗", str(exc))

    @staticmethod
    def _format_transcript_line(text: str, is_final: bool) -> str:
        state = "final" if is_final else "partial"
        return f"[{state}] {text}"

    @classmethod
    def _build_transcript_lines(cls, items) -> list[str]:
        return [cls._format_transcript_line(item.text, item.is_final) for item in items]

    def _report_error(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._error_lock:
            self.recent_errors.append(f"{timestamp} {message}")

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
        asr = FasterWhisperEngine(
            model=self.config.asr.model,
            device=self.config.asr.device,
            compute_type=self.config.asr.compute_type,
            beam_size=self.config.asr.beam_size,
            condition_on_previous_text=self.config.asr.condition_on_previous_text,
            language=self.config.language.meeting_source,
        )
        llm = OllamaClient(
            backend=self.config.llm.backend,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            request_timeout_sec=self.config.llm.request_timeout_sec,
        )
        tts = self._create_tts_engine(self.config.meeting_tts)
        self._health_check_running = True
        self._health_check_started_at = time.monotonic()
        self.local_ai_page.set_runtime_status("health: RUNNING")
        self.diagnostics_page.set_health_summary("health: RUNNING")
        self.diagnostics_page.set_health_details("health check running...")
        self.statusBar().showMessage("health check running...")

        def _worker() -> None:
            try:
                report = run_local_healthcheck(asr_engine=asr, llm_client=llm, tts_engine=tts, warmup=warmup)
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
            summary = "health: FAILED"
            message = f"health check timeout after {int(self._health_check_timeout_sec)}s"
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
            summary = "health: FAILED"
            self.local_ai_page.set_runtime_status(summary)
            self.diagnostics_page.set_health_summary(summary)
            self.diagnostics_page.set_health_details(message)
            self.statusBar().showMessage(summary)
            self._report_error(f"health_check failed: {message}")
            return

        report = payload
        summary = "health: OK" if report.ok else "health: FAILED"
        detail = "\n".join(
            [
                f"ASR: {'OK' if report.asr_ok else 'FAIL'} - {report.asr_message}",
                f"LLM: {'OK' if report.llm_ok else 'FAIL'} - {report.llm_message}",
                f"TTS: {'OK' if report.tts_ok else 'FAIL'} - {report.tts_message}",
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
        self.config.direction.mode = self.audio_routing_page.selected_mode()
        self.live_caption_page.update_config(self.config)
        self.local_ai_page.update_config(self.config)

    def _save_config_to_disk(self) -> Path:
        return save_config(self.config, self.config_path)

    def _build_output_test_audio(self, *, primary_tts: TtsConfig, text: str) -> tuple[np.ndarray, int, str]:
        errors: list[str] = []
        for label, config in self._candidate_test_tts_configs(primary_tts):
            try:
                tts = self._create_tts_engine(config)
                audio = tts.synthesize(text)
                if audio.size == 0:
                    raise ValueError("tts returned empty audio")
                return audio, config.sample_rate, label
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        self._report_error("test_tts_fallback_tone: " + " | ".join(errors[:4]))
        return self._build_test_tone_audio(), 24000, "fallback tone"

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
    def _tts_test_text(language: str) -> str:
        normalized = (language or "").lower()
        if normalized.startswith("zh"):
            return "這是翻譯語音測試。"
        if normalized.startswith("ja"):
            return "これは音声テストです。"
        return "This is a translated speech test."

