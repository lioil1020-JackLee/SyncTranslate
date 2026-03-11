from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.local_ai.ollama_client import OllamaClient
from app.schemas import AppConfig, TtsConfig

_OFFICIAL_FASTER_WHISPER_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
]

_EDGE_VOICE_OPTIONS: list[tuple[str, str]] = [
    ("Chinese Female (TW) - HsiaoChen", "zh-TW-HsiaoChenNeural"),
    ("Chinese Female (TW) - HsiaoYu", "zh-TW-HsiaoYuNeural"),
    ("Chinese Male (TW) - YunJhe", "zh-TW-YunJheNeural"),
    ("English Female (US) - Jenny", "en-US-JennyNeural"),
    ("English Male (US) - Guy", "en-US-GuyNeural"),
    ("English Female (UK) - Sonia", "en-GB-SoniaNeural"),
    ("English Male (UK) - Ryan", "en-GB-RyanNeural"),
]


class PathPickerLineEdit(QLineEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._picker: Callable[[], None] | None = None

    def set_picker(self, picker: Callable[[], None]) -> None:
        self._picker = picker

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self._picker:
            self._picker()


@dataclass(slots=True)
class TtsWidgets:
    group: QGroupBox
    form: QFormLayout
    engine_combo: QComboBox
    edge_voice_combo: QComboBox
    exec_edit: PathPickerLineEdit
    model_edit: PathPickerLineEdit
    config_edit: PathPickerLineEdit
    speaker_combo: QComboBox
    length_scale_spin: QDoubleSpinBox
    noise_scale_spin: QDoubleSpinBox
    noise_w_spin: QDoubleSpinBox
    sample_rate_spin: QSpinBox


class LocalAiPage(QWidget):
    def __init__(
        self,
        on_settings_changed: Callable[[], None] | None,
        on_health_check: Callable[[bool], None],
    ) -> None:
        super().__init__()
        self._on_settings_changed = on_settings_changed
        self._on_health_check = on_health_check
        self._suspend_notify = False
        self._model_load_queue: Queue[tuple[bool, list[str] | str]] = Queue()
        self._model_loading = False

        self.asr_group, asr_form = self._build_asr_group()
        self.llm_group, llm_form = self._build_llm_group()
        self.meeting_tts = self._create_tts_widgets("Speaker Translation Voice")
        self.local_tts = self._create_tts_widgets("Meeting Translation Voice")
        self.runtime_group, runtime_form = self._build_runtime_group()

        self.runtime_status_label = QLabel("Runtime")
        self.runtime_status_label.setWordWrap(True)
        self.runtime_status_label.setStyleSheet("color: #b7bdc6;")

        self.runtime_hint_label = QLabel(
            "Health check and warmup were moved to the Diagnostics tab.\n"
            "This page now focuses on model and runtime settings."
        )
        self.runtime_hint_label.setWordWrap(True)
        self.runtime_hint_label.setStyleSheet("color: #9aa3ad;")

        left_grid = QGridLayout()
        left_grid.setContentsMargins(0, 0, 0, 0)
        left_grid.setHorizontalSpacing(12)
        left_grid.setVerticalSpacing(12)
        left_grid.setColumnStretch(0, 1)
        left_grid.setColumnStretch(1, 1)
        left_grid.addWidget(self.asr_group, 0, 0)
        left_grid.addWidget(self.llm_group, 0, 1)
        left_grid.addWidget(self.meeting_tts.group, 1, 0)
        left_grid.addWidget(self.local_tts.group, 1, 1)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(12)
        left_column.addLayout(left_grid)

        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(10)
        right_column.addWidget(self.runtime_group)
        right_column.addWidget(self.runtime_status_label)
        right_column.addWidget(self.runtime_hint_label)
        right_column.addStretch(1)

        right_panel = QWidget()
        right_panel.setLayout(right_column)
        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(380)
        right_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(14)
        root.addLayout(left_column, 2)
        root.addWidget(right_panel, 1)

        self._normalize_form_label_widths(
            asr_form,
            llm_form,
            self.meeting_tts.form,
            self.local_tts.form,
            runtime_form,
        )

        self._wire_events()

        self._model_poll_timer = QTimer(self)
        self._model_poll_timer.setInterval(150)
        self._model_poll_timer.timeout.connect(self._drain_model_load_queue)
        self._model_poll_timer.start()

        self._reload_tts_speaker_options(self.meeting_tts)
        self._reload_tts_speaker_options(self.local_tts)

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._set_combo_text(self.asr_model_combo, config.asr.model)
            self._select_combo_data(self.asr_device_combo, config.asr.device)
            self.asr_compute_label.setText(self._compute_type_for_device(config.asr.device))
            self.asr_beam_spin.setValue(config.asr.beam_size)
            self.asr_condition_prev_check.setChecked(config.asr.condition_on_previous_text)
            self.asr_partial_interval_spin.setValue(config.asr.streaming.partial_interval_ms)
            self.asr_vad_enabled.setChecked(config.asr.vad.enabled)
            self.asr_min_speech_spin.setValue(config.asr.vad.min_speech_duration_ms)
            self.asr_min_silence_spin.setValue(config.asr.vad.min_silence_duration_ms)
            self.asr_speech_pad_spin.setValue(config.asr.vad.speech_pad_ms)
            self.asr_max_speech_spin.setValue(int(config.asr.vad.max_speech_duration_s))

            self._select_combo_data(self.llm_backend_combo, config.llm.backend)
            self._apply_backend_url(config.llm.backend)
            self._set_combo_text(self.llm_model_combo, config.llm.model)
            self.llm_timeout_spin.setValue(config.llm.request_timeout_sec)
            self.llm_temperature_spin.setValue(config.llm.temperature)
            self.llm_top_p_spin.setValue(config.llm.top_p)
            self.llm_trigger_tokens_spin.setValue(config.llm.sliding_window.trigger_tokens)
            self.llm_context_items_spin.setValue(config.llm.sliding_window.max_context_items)

            self._apply_tts_config(self.meeting_tts, config.meeting_tts)
            self._apply_tts_config(self.local_tts, config.local_tts)

            self.runtime_sample_rate_spin.setValue(config.runtime.sample_rate)
            self.runtime_chunk_spin.setValue(config.runtime.chunk_ms)
            self.runtime_warmup_check.setChecked(config.runtime.warmup_on_start)
            self.runtime_asr_q_spin.setValue(config.runtime.asr_queue_maxsize)
            self.runtime_llm_q_spin.setValue(config.runtime.llm_queue_maxsize)
            self.runtime_tts_q_spin.setValue(config.runtime.tts_queue_maxsize)
        finally:
            self._suspend_notify = False
        self._reload_llm_models()

    def update_config(self, config: AppConfig) -> None:
        config.asr.model = self.asr_model_combo.currentText().strip() or "large-v3"
        config.asr.device = str(self.asr_device_combo.currentData() or "auto")
        config.asr.compute_type = self._compute_type_for_device(config.asr.device)
        config.asr.beam_size = self.asr_beam_spin.value()
        config.asr.condition_on_previous_text = self.asr_condition_prev_check.isChecked()
        config.asr.streaming.partial_interval_ms = self.asr_partial_interval_spin.value()
        config.asr.vad.enabled = self.asr_vad_enabled.isChecked()
        config.asr.vad.min_speech_duration_ms = self.asr_min_speech_spin.value()
        config.asr.vad.min_silence_duration_ms = self.asr_min_silence_spin.value()
        config.asr.vad.speech_pad_ms = self.asr_speech_pad_spin.value()
        config.asr.vad.max_speech_duration_s = float(self.asr_max_speech_spin.value())

        backend = str(self.llm_backend_combo.currentData() or "ollama")
        config.llm.backend = backend
        config.llm.base_url = self._backend_url(backend)
        config.llm.model = self.llm_model_combo.currentText().strip() or "qwen3.5:9b"
        config.llm.request_timeout_sec = self.llm_timeout_spin.value()
        config.llm.temperature = float(self.llm_temperature_spin.value())
        config.llm.top_p = float(self.llm_top_p_spin.value())
        config.llm.sliding_window.trigger_tokens = self.llm_trigger_tokens_spin.value()
        config.llm.sliding_window.max_context_items = self.llm_context_items_spin.value()

        self._update_tts_config(self.meeting_tts, config.meeting_tts)
        self._update_tts_config(self.local_tts, config.local_tts)
        config.tts = config.meeting_tts

        config.runtime.sample_rate = self.runtime_sample_rate_spin.value()
        config.runtime.chunk_ms = self.runtime_chunk_spin.value()
        config.runtime.warmup_on_start = self.runtime_warmup_check.isChecked()
        config.runtime.asr_queue_maxsize = self.runtime_asr_q_spin.value()
        config.runtime.llm_queue_maxsize = self.runtime_llm_q_spin.value()
        config.runtime.tts_queue_maxsize = self.runtime_tts_q_spin.value()

    def set_runtime_status(self, text: str) -> None:
        self.runtime_status_label.setText(text)

    def _build_asr_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.asr_model_combo = QComboBox()
        self.asr_model_combo.setEditable(True)
        for model in self._discover_asr_models():
            self.asr_model_combo.addItem(model)
        self.asr_model_combo.setToolTip("Local models/asr plus faster-whisper official model names")

        self.asr_device_combo = QComboBox()
        self.asr_device_combo.addItem("auto", "auto")
        self.asr_device_combo.addItem("cuda", "cuda")
        self.asr_device_combo.addItem("cpu", "cpu")

        self.asr_compute_label = QLabel("-")
        self.asr_compute_label.setMinimumWidth(160)

        self.asr_beam_spin = QSpinBox()
        self.asr_beam_spin.setRange(1, 8)
        self.asr_condition_prev_check = QCheckBox("Condition on previous text")
        self.asr_partial_interval_spin = QSpinBox()
        self.asr_partial_interval_spin.setRange(200, 5000)
        self.asr_vad_enabled = QCheckBox("Enable VAD")
        self.asr_min_speech_spin = QSpinBox()
        self.asr_min_speech_spin.setRange(100, 4000)
        self.asr_min_silence_spin = QSpinBox()
        self.asr_min_silence_spin.setRange(300, 8000)
        self.asr_speech_pad_spin = QSpinBox()
        self.asr_speech_pad_spin.setRange(0, 2000)
        self.asr_max_speech_spin = QSpinBox()
        self.asr_max_speech_spin.setRange(3, 60)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("Model", self.asr_model_combo)
        form.addRow("Device", self.asr_device_combo)
        form.addRow("Compute", self.asr_compute_label)
        form.addRow("Beam", self.asr_beam_spin)
        form.addRow("", self.asr_condition_prev_check)
        form.addRow("Partial Interval (ms)", self.asr_partial_interval_spin)
        form.addRow("", self.asr_vad_enabled)
        form.addRow("Min Speech (ms)", self.asr_min_speech_spin)
        form.addRow("Min Silence (ms)", self.asr_min_silence_spin)
        form.addRow("Speech Padding (ms)", self.asr_speech_pad_spin)
        form.addRow("Max Speech (s)", self.asr_max_speech_spin)

        group = QGroupBox("ASR (faster-whisper)")
        group.setLayout(form)
        return group, form

    def _build_llm_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.llm_backend_combo = QComboBox()
        self.llm_backend_combo.addItem("Ollama", "ollama")
        self.llm_backend_combo.addItem("LM Studio", "lm_studio")
        self.llm_url_label = QLabel("http://127.0.0.1:11434")
        self.llm_url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        self.llm_reload_models_btn = QPushButton("Reload Models")
        self.llm_timeout_spin = QSpinBox()
        self.llm_timeout_spin.setRange(5, 120)
        self.llm_temperature_spin = QDoubleSpinBox()
        self.llm_temperature_spin.setRange(0.0, 2.0)
        self.llm_temperature_spin.setSingleStep(0.1)
        self.llm_temperature_spin.setDecimals(2)
        self.llm_top_p_spin = QDoubleSpinBox()
        self.llm_top_p_spin.setRange(0.0, 1.0)
        self.llm_top_p_spin.setSingleStep(0.05)
        self.llm_top_p_spin.setDecimals(2)
        self.llm_trigger_tokens_spin = QSpinBox()
        self.llm_trigger_tokens_spin.setRange(5, 120)
        self.llm_context_items_spin = QSpinBox()
        self.llm_context_items_spin.setRange(2, 20)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("Backend", self.llm_backend_combo)
        form.addRow("Server URL", self.llm_url_label)
        form.addRow("Model", self.llm_model_combo)
        form.addRow("", self.llm_reload_models_btn)
        form.addRow("Timeout (s)", self.llm_timeout_spin)
        form.addRow("Temperature", self.llm_temperature_spin)
        form.addRow("Top P", self.llm_top_p_spin)
        form.addRow("Trigger Token", self.llm_trigger_tokens_spin)
        form.addRow("Context Items", self.llm_context_items_spin)

        group = QGroupBox("LLM")
        group.setLayout(form)
        return group, form

    def _build_runtime_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.runtime_sample_rate_spin = QSpinBox()
        self.runtime_sample_rate_spin.setRange(8000, 192000)
        self.runtime_chunk_spin = QSpinBox()
        self.runtime_chunk_spin.setRange(20, 1000)
        self.runtime_warmup_check = QCheckBox("Warmup on Start")
        self.runtime_asr_q_spin = QSpinBox()
        self.runtime_asr_q_spin.setRange(8, 512)
        self.runtime_llm_q_spin = QSpinBox()
        self.runtime_llm_q_spin.setRange(8, 512)
        self.runtime_tts_q_spin = QSpinBox()
        self.runtime_tts_q_spin.setRange(8, 512)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("Sample Rate", self.runtime_sample_rate_spin)
        form.addRow("Chunk (ms)", self.runtime_chunk_spin)
        form.addRow("", self.runtime_warmup_check)
        form.addRow("ASR Queue", self.runtime_asr_q_spin)
        form.addRow("LLM Queue", self.runtime_llm_q_spin)
        form.addRow("TTS Queue", self.runtime_tts_q_spin)

        group = QGroupBox("Runtime")
        group.setLayout(form)
        return group, form

    def _create_tts_widgets(self, title: str) -> TtsWidgets:
        engine_combo = QComboBox()
        engine_combo.addItem("Piper (Local)", "piper")
        engine_combo.addItem("Edge TTS", "edge_tts")

        edge_voice_combo = QComboBox()
        for label, voice_name in _EDGE_VOICE_OPTIONS:
            edge_voice_combo.addItem(label, voice_name)

        exec_edit = PathPickerLineEdit()
        model_edit = PathPickerLineEdit()
        config_edit = PathPickerLineEdit()
        speaker_combo = QComboBox()
        length_scale_spin = QDoubleSpinBox()
        length_scale_spin.setRange(0.1, 3.0)
        length_scale_spin.setSingleStep(0.05)
        length_scale_spin.setDecimals(2)
        noise_scale_spin = QDoubleSpinBox()
        noise_scale_spin.setRange(0.0, 2.0)
        noise_scale_spin.setSingleStep(0.05)
        noise_scale_spin.setDecimals(3)
        noise_w_spin = QDoubleSpinBox()
        noise_w_spin.setRange(0.0, 2.0)
        noise_w_spin.setSingleStep(0.05)
        noise_w_spin.setDecimals(3)
        sample_rate_spin = QSpinBox()
        sample_rate_spin.setRange(8000, 48000)

        for editor in (exec_edit, model_edit, config_edit):
            editor.setReadOnly(True)
            editor.setCursor(Qt.CursorShape.PointingHandCursor)
            editor.setToolTip("Click to pick a file")
            self._set_uniform_field_style(editor, minimum_width=180)

        for compact in (
            engine_combo,
            edge_voice_combo,
            speaker_combo,
            length_scale_spin,
            noise_scale_spin,
            noise_w_spin,
            sample_rate_spin,
        ):
            self._set_uniform_field_style(compact, minimum_width=180)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("Engine", engine_combo)
        form.addRow("Edge Voice", edge_voice_combo)
        form.addRow("Piper Executable", exec_edit)
        form.addRow("Model Path", model_edit)
        form.addRow("Config Path", config_edit)
        form.addRow("Piper Speaker", speaker_combo)
        form.addRow("Length Scale", length_scale_spin)
        form.addRow("Noise Scale", noise_scale_spin)
        form.addRow("Noise W", noise_w_spin)
        form.addRow("Sample Rate", sample_rate_spin)

        group = QGroupBox(title)
        group.setLayout(form)
        return TtsWidgets(
            group=group,
            form=form,
            engine_combo=engine_combo,
            edge_voice_combo=edge_voice_combo,
            exec_edit=exec_edit,
            model_edit=model_edit,
            config_edit=config_edit,
            speaker_combo=speaker_combo,
            length_scale_spin=length_scale_spin,
            noise_scale_spin=noise_scale_spin,
            noise_w_spin=noise_w_spin,
            sample_rate_spin=sample_rate_spin,
        )

    def _wire_events(self) -> None:
        for compact in (
            self.asr_model_combo,
            self.asr_device_combo,
            self.asr_beam_spin,
            self.asr_partial_interval_spin,
            self.asr_min_speech_spin,
            self.asr_min_silence_spin,
            self.asr_speech_pad_spin,
            self.asr_max_speech_spin,
            self.llm_backend_combo,
            self.llm_model_combo,
            self.llm_timeout_spin,
            self.llm_temperature_spin,
            self.llm_top_p_spin,
            self.llm_trigger_tokens_spin,
            self.llm_context_items_spin,
            self.runtime_sample_rate_spin,
            self.runtime_chunk_spin,
            self.runtime_asr_q_spin,
            self.runtime_llm_q_spin,
            self.runtime_tts_q_spin,
        ):
            self._set_uniform_field_style(compact, minimum_width=180)

        self.llm_url_label.setMinimumWidth(180)
        self.llm_backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.asr_device_combo.currentIndexChanged.connect(self._on_asr_device_changed)
        self.llm_reload_models_btn.clicked.connect(self._reload_llm_models)

        for widget in (
            self.asr_model_combo,
            self.asr_device_combo,
            self.asr_beam_spin,
            self.asr_partial_interval_spin,
            self.asr_vad_enabled,
            self.asr_condition_prev_check,
            self.asr_min_speech_spin,
            self.asr_min_silence_spin,
            self.asr_speech_pad_spin,
            self.asr_max_speech_spin,
            self.llm_backend_combo,
            self.llm_model_combo,
            self.llm_timeout_spin,
            self.llm_temperature_spin,
            self.llm_top_p_spin,
            self.llm_trigger_tokens_spin,
            self.llm_context_items_spin,
            self.runtime_sample_rate_spin,
            self.runtime_chunk_spin,
            self.runtime_warmup_check,
            self.runtime_asr_q_spin,
            self.runtime_llm_q_spin,
            self.runtime_tts_q_spin,
        ):
            self._connect_change_signal(widget)

        self._bind_tts_widget_events(self.meeting_tts, "speaker")
        self._bind_tts_widget_events(self.local_tts, "meeting")

    def _bind_tts_widget_events(self, widgets: TtsWidgets, title: str) -> None:
        widgets.exec_edit.set_picker(lambda: self._pick_path(widgets.exec_edit, f"Select {title} Piper executable", True))
        widgets.model_edit.set_picker(lambda: self._pick_path(widgets.model_edit, f"Select {title} TTS model", False))
        widgets.config_edit.set_picker(lambda: self._pick_path(widgets.config_edit, f"Select {title} TTS config", False))
        widgets.config_edit.textChanged.connect(lambda *_: self._reload_tts_speaker_options(widgets))
        for widget in (
            widgets.engine_combo,
            widgets.edge_voice_combo,
            widgets.exec_edit,
            widgets.model_edit,
            widgets.config_edit,
            widgets.speaker_combo,
            widgets.length_scale_spin,
            widgets.noise_scale_spin,
            widgets.noise_w_spin,
            widgets.sample_rate_spin,
        ):
            self._connect_change_signal(widget)

    def _apply_tts_config(self, widgets: TtsWidgets, config: TtsConfig) -> None:
        self._select_combo_data(widgets.engine_combo, config.engine)
        self._select_combo_data(widgets.edge_voice_combo, config.voice_name.strip() or "zh-TW-HsiaoChenNeural")
        widgets.exec_edit.setText(config.executable_path)
        widgets.model_edit.setText(config.model_path)
        widgets.config_edit.setText(config.config_path)
        widgets.length_scale_spin.setValue(config.length_scale)
        widgets.noise_scale_spin.setValue(config.noise_scale)
        widgets.noise_w_spin.setValue(config.noise_w)
        widgets.sample_rate_spin.setValue(config.sample_rate)
        self._reload_tts_speaker_options(widgets, selected_id=config.speaker_id)

    @staticmethod
    def _update_tts_config(widgets: TtsWidgets, config: TtsConfig) -> None:
        config.engine = str(widgets.engine_combo.currentData() or "piper")
        config.voice_name = str(widgets.edge_voice_combo.currentData() or "")
        config.executable_path = widgets.exec_edit.text().strip()
        config.model_path = widgets.model_edit.text().strip()
        config.config_path = widgets.config_edit.text().strip()
        speaker_id = widgets.speaker_combo.currentData()
        config.speaker_id = int(speaker_id) if speaker_id is not None else 0
        config.length_scale = float(widgets.length_scale_spin.value())
        config.noise_scale = float(widgets.noise_scale_spin.value())
        config.noise_w = float(widgets.noise_w_spin.value())
        config.sample_rate = widgets.sample_rate_spin.value()

    def _reload_tts_speaker_options(self, widgets: TtsWidgets, selected_id: int | None = None) -> None:
        speaker_rows = self._load_speaker_rows_from_tts_config(widgets.config_edit.text().strip())
        if not speaker_rows:
            speaker_rows = [("Default Speaker", 0)]

        current_id = selected_id
        if current_id is None and widgets.speaker_combo.currentData() is not None:
            current_id = int(widgets.speaker_combo.currentData())
        if current_id is None:
            current_id = 0

        widgets.speaker_combo.blockSignals(True)
        widgets.speaker_combo.clear()
        for label, sid in speaker_rows:
            widgets.speaker_combo.addItem(label, sid)
        widgets.speaker_combo.blockSignals(False)

        idx = widgets.speaker_combo.findData(current_id)
        widgets.speaker_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _load_speaker_rows_from_tts_config(self, config_path_text: str) -> list[tuple[str, int]]:
        path = self._resolve_path(config_path_text)
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

        speaker_map = data.get("speaker_id_map")
        if isinstance(speaker_map, dict):
            rows: list[tuple[str, int]] = []
            for name, sid in speaker_map.items():
                try:
                    rows.append((f"Speaker: {name}", int(sid)))
                except Exception:
                    continue
            rows.sort(key=lambda row: row[1])
            return rows

        speakers = data.get("speakers")
        if isinstance(speakers, list):
            rows = []
            for idx, item in enumerate(speakers):
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("speaker") or f"Speaker {idx + 1}")
                else:
                    name = str(item)
                rows.append((f"Speaker: {name}", idx))
            return rows

        return []

    def _reload_llm_models(self) -> None:
        if self._model_loading:
            return
        backend = str(self.llm_backend_combo.currentData() or "ollama")
        url = self._backend_url(backend)
        self._model_loading = True
        self.llm_reload_models_btn.setEnabled(False)
        self.llm_reload_models_btn.setText("Loading...")

        def _worker() -> None:
            try:
                client = OllamaClient(backend=backend, base_url=url, model="", request_timeout_sec=3.0)
                raw_models = [model for model in client.list_models() if model.strip()]
                self._model_load_queue.put((True, self._filter_llm_models(raw_models)))
            except Exception as exc:
                self._model_load_queue.put((False, str(exc)))

        Thread(target=_worker, daemon=True).start()

    def _drain_model_load_queue(self) -> None:
        try:
            ok, payload = self._model_load_queue.get_nowait()
        except Empty:
            return

        self._model_loading = False
        self.llm_reload_models_btn.setEnabled(True)
        self.llm_reload_models_btn.setText("Reload Models")

        if not ok:
            self.runtime_status_label.setText(f"LLM model load failed: {payload}")
            return

        current = self.llm_model_combo.currentText().strip()
        models = payload if isinstance(payload, list) else []
        if not models:
            self.runtime_status_label.setText("No LLM models found")
            self.llm_model_combo.clear()
            return

        self.llm_model_combo.blockSignals(True)
        self.llm_model_combo.clear()
        for model in models:
            self.llm_model_combo.addItem(model)
        self.llm_model_combo.blockSignals(False)

        if current and current in models:
            self._set_combo_text(self.llm_model_combo, current)
        else:
            self._set_combo_text(self.llm_model_combo, models[0])
        self._notify_settings_changed()

    def _on_backend_changed(self) -> None:
        backend = str(self.llm_backend_combo.currentData() or "ollama")
        self._apply_backend_url(backend)
        self._reload_llm_models()
        self._notify_settings_changed()

    def _on_asr_device_changed(self) -> None:
        device = str(self.asr_device_combo.currentData() or "auto")
        self.asr_compute_label.setText(self._compute_type_for_device(device))
        self._notify_settings_changed()

    def _notify_settings_changed(self) -> None:
        if self._suspend_notify:
            return
        if self._on_settings_changed:
            self._on_settings_changed()

    def _pick_path(self, target: QLineEdit, title: str, exe_only: bool) -> None:
        current = target.text().strip() or "."
        start_dir = str(Path(current).resolve()) if current else "."
        if exe_only:
            selected, _ = QFileDialog.getOpenFileName(self, title, start_dir, "Executable (*.exe);;All Files (*)")
        else:
            selected, _ = QFileDialog.getOpenFileName(self, title, start_dir, "All Files (*)")
        if not selected:
            return
        target.setText(self._to_project_relative(selected))
        self._notify_settings_changed()

    @staticmethod
    def _configure_form_layout(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

    @staticmethod
    def _set_uniform_field_style(widget: QWidget, minimum_width: int = 160) -> None:
        widget.setMinimumWidth(minimum_width)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _normalize_form_label_widths(*forms: QFormLayout) -> None:
        max_width = 0
        labels: list[QWidget] = []
        for form in forms:
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item is None or item.widget() is None:
                    continue
                labels.append(item.widget())
                max_width = max(max_width, item.widget().sizeHint().width())
        for label in labels:
            label.setMinimumWidth(max_width)

    @staticmethod
    def _compute_type_for_device(device: str) -> str:
        if device == "cpu":
            return "int8"
        if device == "cuda":
            return "float16"
        return "auto"

    def _apply_backend_url(self, backend: str) -> None:
        self.llm_url_label.setText(self._backend_url(backend))

    @staticmethod
    def _backend_url(backend: str) -> str:
        return "http://127.0.0.1:1234" if backend == "lm_studio" else "http://127.0.0.1:11434"

    @staticmethod
    def _filter_llm_models(models: list[str]) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        for model in models:
            value = model.strip()
            lower = value.lower()
            if not value or lower in seen:
                continue
            if any(token in lower for token in ("embed", "embedding", "nomic-embed")):
                continue
            seen.add(lower)
            filtered.append(value)
        return filtered

    @staticmethod
    def _discover_asr_models() -> list[str]:
        model_root = Path("models") / "asr"
        local_models: list[str] = []
        if model_root.exists():
            for path in sorted(model_root.iterdir()):
                if path.is_dir():
                    local_models.append(path.name)
                elif path.suffix.lower() in {".bin", ".gguf", ".pt"}:
                    local_models.append(path.stem)

        merged: list[str] = []
        seen: set[str] = set()
        for name in local_models + _OFFICIAL_FASTER_WHISPER_MODELS:
            if name and name not in seen:
                seen.add(name)
                merged.append(name)
        return merged

    def _connect_change_signal(self, widget: QWidget) -> None:
        if hasattr(widget, "textChanged"):
            widget.textChanged.connect(lambda *_: self._notify_settings_changed())  # type: ignore[attr-defined]
        if hasattr(widget, "currentTextChanged"):
            widget.currentTextChanged.connect(lambda *_: self._notify_settings_changed())  # type: ignore[attr-defined]
        if hasattr(widget, "valueChanged"):
            widget.valueChanged.connect(lambda *_: self._notify_settings_changed())  # type: ignore[attr-defined]
        if hasattr(widget, "toggled"):
            widget.toggled.connect(lambda *_: self._notify_settings_changed())  # type: ignore[attr-defined]

    @staticmethod
    def _to_project_relative(abs_path: str) -> str:
        path = Path(abs_path).resolve()
        cwd = Path.cwd().resolve()
        try:
            rel = path.relative_to(cwd)
            return ".\\" + str(rel).replace("/", "\\")
        except Exception:
            return str(path).replace("/", "\\")

    @staticmethod
    def _resolve_path(path_text: str) -> Path | None:
        if not path_text:
            return None
        path = Path(path_text)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable():
            combo.setEditText(text)
