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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.local_ai.lm_studio_client import LmStudioClient
from app.schemas import AppConfig, TtsChannelOverride, TtsConfig, merge_tts_configs

_OFFICIAL_FASTER_WHISPER_MODELS = [
    # 多語言模型（由大到小）
    "large-v3",
    "large-v3-turbo",
    "large-v2",
    "large-v1",
    "medium",
    "small",
    "base",
    "tiny",
    # 英文專用模型
    "medium.en",
    "small.en",
    "base.en",
    "tiny.en",
    # Distil 輕量縮小版
    "distil-large-v3",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
]

# 常用取樣率選項（顯示用與資料用）
_COMMON_SAMPLE_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000]

_EDGE_VOICE_OPTIONS: list[tuple[str, str]] = [
    ("中文女聲（台灣）- HsiaoChen", "zh-TW-HsiaoChenNeural"),
    ("中文女聲（台灣）- HsiaoYu", "zh-TW-HsiaoYuNeural"),
    ("中文男聲（台灣）- YunJhe", "zh-TW-YunJheNeural"),
    ("英文女聲（美國）- Jenny", "en-US-JennyNeural"),
    ("英文男聲（美國）- Guy", "en-US-GuyNeural"),
    ("英文女聲（英國）- Sonia", "en-GB-SoniaNeural"),
    ("英文男聲（英國）- Ryan", "en-GB-RyanNeural"),
]

_CONTROL_HEIGHT = 26
_FORM_V_SPACING = 4
# reduce extra group height to free vertical space
_GROUP_EXTRA_HEIGHT = 18
_GROUP_ROW_SPACING = 8
# reduce page padding to allow slightly more content
_PAGE_MIN_PADDING = 6


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
    sample_rate_spin: QComboBox


@dataclass(slots=True)
class TtsChannelWidgets:
    group: QGroupBox
    form: QFormLayout
    engine_combo: QComboBox
    edge_voice_combo: QComboBox
    sample_rate_spin: QComboBox
    noise_w_spin: QDoubleSpinBox


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
        self._preferred_llm_model = ""
        self.setStyleSheet("QWidget { font-size: 10pt; }")

        self.asr_group, asr_form = self._build_asr_group()
        self.llm_group, llm_form = self._build_llm_group()
        self.runtime_group, runtime_form = self._build_runtime_group()
        self.base_tts = self._create_tts_widgets("TTS 模型設定")
        self.local_tts_override = self._create_tts_channel_widgets("中文輸出通道覆寫", "zh")
        self.remote_tts_override = self._create_tts_channel_widgets("英文輸出通道覆寫", "en")
        self.tts_channel_group = self._build_tts_channel_compact_group()

        self.runtime_status_label = QLabel("執行狀態")
        self.runtime_status_label.setWordWrap(True)
        self.runtime_status_label.setStyleSheet("color: #b7bdc6;")

        # Keep two independent columns so one side's height won't squeeze/expand the other.
        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(8)
        left_column.addWidget(self.asr_group)
        left_column.addWidget(self.runtime_group)
        left_column.addStretch(1)

        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(8)
        right_column.addWidget(self.llm_group)
        right_column.addWidget(self.base_tts.group)
        right_column.addWidget(self.tts_channel_group)
        right_column.addStretch(1)

        top_columns = QHBoxLayout()
        top_columns.setContentsMargins(0, 0, 0, 0)
        top_columns.setSpacing(8)
        top_columns.addLayout(left_column, 1)
        top_columns.addLayout(right_column, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(_PAGE_MIN_PADDING, _PAGE_MIN_PADDING, _PAGE_MIN_PADDING, _PAGE_MIN_PADDING)
        root.setSpacing(6)
        root.addLayout(top_columns)
        root.addStretch(1)

        self._normalize_form_label_widths(
            asr_form,
            llm_form,
            self.base_tts.form,
            runtime_form,
        )
        # Let layout use natural heights to avoid visual compression and large empty blocks.

        self._wire_events()

        self._model_poll_timer = QTimer(self)
        self._model_poll_timer.setInterval(150)
        self._model_poll_timer.timeout.connect(self._drain_model_load_queue)
        self._model_poll_timer.start()

        self._reload_tts_speaker_options(self.base_tts)

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            zh_asr = config.asr_channels.local
            en_asr = config.asr_channels.remote
            self._select_combo_data(self.asr_engine_combo, zh_asr.engine)
            self._set_combo_text(self.asr_model_combo, zh_asr.model)
            self._set_combo_text(self.remote_asr_model_combo, en_asr.model)
            self._select_combo_data(self.asr_device_combo, zh_asr.device)
            self.asr_compute_label.setText(self._compute_type_for_device(zh_asr.device))
            self.asr_beam_spin.setValue(zh_asr.beam_size)
            self.remote_asr_beam_spin.setValue(en_asr.beam_size)
            self.asr_condition_prev_check.setChecked(zh_asr.condition_on_previous_text)
            self.remote_asr_condition_prev_check.setChecked(en_asr.condition_on_previous_text)
            self.asr_temperature_fallback_local_edit.setText(zh_asr.temperature_fallback)
            self.asr_temperature_fallback_remote_edit.setText(en_asr.temperature_fallback)
            self.asr_partial_interval_spin.setValue(zh_asr.streaming.partial_interval_ms)
            self.remote_asr_partial_interval_spin.setValue(en_asr.streaming.partial_interval_ms)
            self.asr_partial_history_spin.setValue(zh_asr.streaming.partial_history_seconds)
            self.remote_asr_partial_history_spin.setValue(en_asr.streaming.partial_history_seconds)
            self.asr_final_history_spin.setValue(zh_asr.streaming.final_history_seconds)
            self.remote_asr_final_history_spin.setValue(en_asr.streaming.final_history_seconds)
            self.asr_vad_enabled.setChecked(zh_asr.vad.enabled)
            self.remote_asr_vad_enabled.setChecked(en_asr.vad.enabled)
            self.asr_min_speech_spin.setValue(zh_asr.vad.min_speech_duration_ms)
            self.remote_asr_min_speech_spin.setValue(en_asr.vad.min_speech_duration_ms)
            self.asr_min_silence_spin.setValue(zh_asr.vad.min_silence_duration_ms)
            self.remote_asr_min_silence_spin.setValue(en_asr.vad.min_silence_duration_ms)
            self.asr_speech_pad_spin.setValue(zh_asr.vad.speech_pad_ms)
            self.remote_asr_speech_pad_spin.setValue(en_asr.vad.speech_pad_ms)
            self.asr_max_speech_spin.setValue(int(zh_asr.vad.max_speech_duration_s))
            self.remote_asr_max_speech_spin.setValue(int(en_asr.vad.max_speech_duration_s))
            self.asr_rms_threshold_spin.setValue(zh_asr.vad.rms_threshold)
            self.remote_asr_rms_threshold_spin.setValue(en_asr.vad.rms_threshold)
            self.asr_no_speech_threshold_spin.setValue(zh_asr.no_speech_threshold)
            self.remote_asr_no_speech_threshold_spin.setValue(en_asr.no_speech_threshold)
            self.asr_queue_local_spin.setValue(
                int(getattr(config.runtime, "asr_queue_maxsize_local", config.runtime.asr_queue_maxsize))
            )
            self.asr_queue_remote_spin.setValue(
                int(getattr(config.runtime, "asr_queue_maxsize_remote", config.runtime.asr_queue_maxsize))
            )

            zh_en_llm = config.llm_channels.local
            en_zh_llm = config.llm_channels.remote
            self._select_combo_data(self.llm_backend_combo, zh_en_llm.backend)
            self._apply_backend_url(zh_en_llm.backend, self.llm_url_label)
            self._preferred_llm_model = zh_en_llm.model.strip()
            self._set_combo_text(self.llm_model_combo, zh_en_llm.model)
            self._set_combo_text(self.remote_llm_model_combo, en_zh_llm.model)
            self.llm_timeout_spin.setValue(zh_en_llm.request_timeout_sec)
            self.remote_llm_timeout_spin.setValue(en_zh_llm.request_timeout_sec)
            self.llm_temperature_spin.setValue(zh_en_llm.temperature)
            self.remote_llm_temperature_spin.setValue(en_zh_llm.temperature)
            self.llm_top_p_spin.setValue(zh_en_llm.top_p)
            self.remote_llm_top_p_spin.setValue(en_zh_llm.top_p)
            self.llm_max_tokens_spin.setValue(zh_en_llm.max_output_tokens)
            self.remote_llm_max_tokens_spin.setValue(en_zh_llm.max_output_tokens)
            self.llm_repeat_penalty_spin.setValue(zh_en_llm.repeat_penalty)
            self.remote_llm_repeat_penalty_spin.setValue(en_zh_llm.repeat_penalty)
            self.llm_stop_tokens_edit.setPlainText(zh_en_llm.stop_tokens)
            self.remote_llm_stop_tokens_edit.setPlainText(en_zh_llm.stop_tokens)
            self.llm_sliding_window_enabled.setChecked(zh_en_llm.sliding_window.enabled)
            self.remote_llm_sliding_window_enabled.setChecked(en_zh_llm.sliding_window.enabled)
            self.llm_trigger_tokens_spin.setValue(zh_en_llm.sliding_window.trigger_tokens)
            self.remote_llm_trigger_tokens_spin.setValue(en_zh_llm.sliding_window.trigger_tokens)
            self.llm_context_items_spin.setValue(zh_en_llm.sliding_window.max_context_items)
            self.remote_llm_context_items_spin.setValue(en_zh_llm.sliding_window.max_context_items)
            self.llm_queue_local_spin.setValue(
                int(getattr(config.runtime, "llm_queue_maxsize_local", config.runtime.llm_queue_maxsize))
            )
            self.llm_queue_remote_spin.setValue(
                int(getattr(config.runtime, "llm_queue_maxsize_remote", config.runtime.llm_queue_maxsize))
            )

            self._apply_tts_config(self.base_tts, config.tts)
            self._apply_tts_override(self.local_tts_override, config.tts_channels.local, fallback=config.meeting_tts)
            self._apply_tts_override(self.remote_tts_override, config.tts_channels.remote, fallback=config.local_tts)
            self.tts_queue_local_spin.setValue(
                int(getattr(config.runtime, "tts_queue_maxsize_local", config.runtime.tts_queue_maxsize))
            )
            self.tts_queue_remote_spin.setValue(
                int(getattr(config.runtime, "tts_queue_maxsize_remote", config.runtime.tts_queue_maxsize))
            )

            self._select_combo_data(self.runtime_sample_rate_spin, config.runtime.sample_rate)
            self.runtime_chunk_spin.setValue(config.runtime.chunk_ms)
            self.runtime_asr_pre_roll_spin.setValue(int(getattr(config.runtime, "asr_pre_roll_ms", 500)))
            self.runtime_tts_cancel_pending_check.setChecked(config.runtime.tts_cancel_pending_on_new_final)
            self._select_combo_data(self.runtime_tts_cancel_policy_combo, str(getattr(config.runtime, "tts_cancel_policy", "all_pending")))
            self.runtime_tts_max_wait_spin.setValue(int(getattr(config.runtime, "tts_max_wait_ms", 4000)))
            self.runtime_tts_max_chars_spin.setValue(int(getattr(config.runtime, "tts_max_chars", 200)))
            self.runtime_llm_streaming_tokens_spin.setValue(int(getattr(config.runtime, "llm_streaming_tokens", 16)))
            self.runtime_max_pipeline_latency_spin.setValue(int(getattr(config.runtime, "max_pipeline_latency_ms", 3000)))
            self.runtime_translation_cache_spin.setValue(config.runtime.translation_exact_cache_size)
            self.runtime_prefix_delta_spin.setValue(config.runtime.translation_prefix_min_delta_chars)
            self.runtime_tts_drop_threshold_spin.setValue(int(getattr(config.runtime, "tts_drop_backlog_threshold", 6)))
            self.runtime_local_echo_guard_enabled_check.setChecked(config.runtime.local_echo_guard_enabled)
            self.runtime_local_echo_guard_resume_delay_spin.setValue(config.runtime.local_echo_guard_resume_delay_ms)
            self.runtime_remote_echo_guard_resume_delay_spin.setValue(config.runtime.remote_echo_guard_resume_delay_ms)
        finally:
            self._suspend_notify = False
        self._reload_llm_models()

    def update_config(self, config: AppConfig) -> None:
        zh_asr = config.asr_channels.local
        en_asr = config.asr_channels.remote
        shared_engine = str(self.asr_engine_combo.currentData() or "faster_whisper")
        shared_device = str(self.asr_device_combo.currentData() or "cuda")
        shared_compute = self._compute_type_for_device(shared_device)
        zh_asr.engine = shared_engine
        zh_asr.model = self.asr_model_combo.currentText().strip() or "large-v3"
        zh_asr.device = shared_device
        zh_asr.compute_type = shared_compute
        zh_asr.beam_size = self.asr_beam_spin.value()
        zh_asr.condition_on_previous_text = self.asr_condition_prev_check.isChecked()
        zh_asr.temperature_fallback = self.asr_temperature_fallback_local_edit.text().strip() or "0.0,0.2"
        zh_asr.streaming.partial_interval_ms = self.asr_partial_interval_spin.value()
        zh_asr.streaming.partial_history_seconds = self.asr_partial_history_spin.value()
        zh_asr.streaming.final_history_seconds = self.asr_final_history_spin.value()
        zh_asr.vad.enabled = self.asr_vad_enabled.isChecked()
        zh_asr.vad.min_speech_duration_ms = self.asr_min_speech_spin.value()
        zh_asr.vad.min_silence_duration_ms = self.asr_min_silence_spin.value()
        zh_asr.vad.speech_pad_ms = self.asr_speech_pad_spin.value()
        zh_asr.vad.max_speech_duration_s = float(self.asr_max_speech_spin.value())
        zh_asr.vad.rms_threshold = float(self.asr_rms_threshold_spin.value())
        zh_asr.no_speech_threshold = float(self.asr_no_speech_threshold_spin.value())
        config.runtime.asr_queue_maxsize_local = self.asr_queue_local_spin.value()

        en_asr.engine = shared_engine
        en_asr.model = self.remote_asr_model_combo.currentText().strip() or "large-v3"
        en_asr.device = shared_device
        en_asr.compute_type = shared_compute
        en_asr.beam_size = self.remote_asr_beam_spin.value()
        en_asr.condition_on_previous_text = self.remote_asr_condition_prev_check.isChecked()
        en_asr.temperature_fallback = self.asr_temperature_fallback_remote_edit.text().strip() or "0.0,0.2,0.4"
        en_asr.streaming.partial_interval_ms = self.remote_asr_partial_interval_spin.value()
        en_asr.streaming.partial_history_seconds = self.remote_asr_partial_history_spin.value()
        en_asr.streaming.final_history_seconds = self.remote_asr_final_history_spin.value()
        en_asr.vad.enabled = self.remote_asr_vad_enabled.isChecked()
        en_asr.vad.min_speech_duration_ms = self.remote_asr_min_speech_spin.value()
        en_asr.vad.min_silence_duration_ms = self.remote_asr_min_silence_spin.value()
        en_asr.vad.speech_pad_ms = self.remote_asr_speech_pad_spin.value()
        en_asr.vad.max_speech_duration_s = float(self.remote_asr_max_speech_spin.value())
        en_asr.vad.rms_threshold = float(self.remote_asr_rms_threshold_spin.value())
        en_asr.no_speech_threshold = float(self.remote_asr_no_speech_threshold_spin.value())
        config.runtime.asr_queue_maxsize_remote = self.asr_queue_remote_spin.value()
        config.asr = zh_asr

        backend = str(self.llm_backend_combo.currentData() or "lm_studio")
        zh_en_llm = config.llm_channels.local
        en_zh_llm = config.llm_channels.remote
        shared_url = self._backend_url(backend)
        for llm_cfg, model_text in (
            (zh_en_llm, self.llm_model_combo.currentText().strip()),
            (en_zh_llm, self.remote_llm_model_combo.currentText().strip()),
        ):
            llm_cfg.backend = backend
            llm_cfg.base_url = shared_url
            llm_cfg.model = model_text or "qwen/qwen3.5-9b"
        zh_en_llm.request_timeout_sec = self.llm_timeout_spin.value()
        en_zh_llm.request_timeout_sec = self.remote_llm_timeout_spin.value()
        zh_en_llm.temperature = float(self.llm_temperature_spin.value())
        en_zh_llm.temperature = float(self.remote_llm_temperature_spin.value())
        zh_en_llm.top_p = float(self.llm_top_p_spin.value())
        en_zh_llm.top_p = float(self.remote_llm_top_p_spin.value())
        zh_en_llm.max_output_tokens = self.llm_max_tokens_spin.value()
        en_zh_llm.max_output_tokens = self.remote_llm_max_tokens_spin.value()
        zh_en_llm.repeat_penalty = float(self.llm_repeat_penalty_spin.value())
        en_zh_llm.repeat_penalty = float(self.remote_llm_repeat_penalty_spin.value())
        zh_en_llm.stop_tokens = self.llm_stop_tokens_edit.toPlainText().strip()
        en_zh_llm.stop_tokens = self.remote_llm_stop_tokens_edit.toPlainText().strip()
        zh_en_llm.sliding_window.enabled = self.llm_sliding_window_enabled.isChecked()
        en_zh_llm.sliding_window.enabled = self.remote_llm_sliding_window_enabled.isChecked()
        zh_en_llm.sliding_window.trigger_tokens = self.llm_trigger_tokens_spin.value()
        en_zh_llm.sliding_window.trigger_tokens = self.remote_llm_trigger_tokens_spin.value()
        zh_en_llm.sliding_window.max_context_items = self.llm_context_items_spin.value()
        en_zh_llm.sliding_window.max_context_items = self.remote_llm_context_items_spin.value()
        config.runtime.llm_queue_maxsize_local = self.llm_queue_local_spin.value()
        config.runtime.llm_queue_maxsize_remote = self.llm_queue_remote_spin.value()

        chosen_model = self.llm_model_combo.currentText().strip()
        if not chosen_model:
            chosen_model = (self._preferred_llm_model or "").strip()
        if not chosen_model:
            chosen_model = (zh_en_llm.model or "").strip()
        zh_en_llm.model = chosen_model or "qwen/qwen3.5-9b"
        if not (en_zh_llm.model or "").strip():
            en_zh_llm.model = zh_en_llm.model
        self._preferred_llm_model = zh_en_llm.model
        config.llm = zh_en_llm

        self._update_tts_config(self.base_tts, config.tts)
        config.tts_channels.local = self._update_tts_override(self.local_tts_override)
        config.tts_channels.remote = self._update_tts_override(self.remote_tts_override)
        config.meeting_tts = merge_tts_configs(config.tts, config.meeting_tts, config.tts_channels.local)
        config.local_tts = merge_tts_configs(config.tts, config.local_tts, config.tts_channels.remote)
        config.runtime.tts_queue_maxsize_local = self.tts_queue_local_spin.value()
        config.runtime.tts_queue_maxsize_remote = self.tts_queue_remote_spin.value()

        try:
            config.runtime.sample_rate = int(self.runtime_sample_rate_spin.currentData() or config.runtime.sample_rate)
        except Exception:
            config.runtime.sample_rate = int(str(self.runtime_sample_rate_spin.currentText()).strip() or config.runtime.sample_rate)
        config.runtime.chunk_ms = self.runtime_chunk_spin.value()
        config.runtime.asr_pre_roll_ms = self.runtime_asr_pre_roll_spin.value()
        config.runtime.tts_cancel_pending_on_new_final = self.runtime_tts_cancel_pending_check.isChecked()
        config.runtime.tts_cancel_policy = str(self.runtime_tts_cancel_policy_combo.currentData() or "all_pending")
        config.runtime.tts_max_wait_ms = self.runtime_tts_max_wait_spin.value()
        config.runtime.tts_max_chars = self.runtime_tts_max_chars_spin.value()
        config.runtime.llm_streaming_tokens = self.runtime_llm_streaming_tokens_spin.value()
        config.runtime.max_pipeline_latency_ms = self.runtime_max_pipeline_latency_spin.value()
        # Keep legacy shared fields aligned for backward compatibility.
        config.runtime.asr_queue_maxsize = config.runtime.asr_queue_maxsize_local
        config.runtime.llm_queue_maxsize = config.runtime.llm_queue_maxsize_local
        config.runtime.tts_queue_maxsize = config.runtime.tts_queue_maxsize_local
        config.runtime.local_echo_guard_enabled = self.runtime_local_echo_guard_enabled_check.isChecked()
        config.runtime.local_echo_guard_resume_delay_ms = self.runtime_local_echo_guard_resume_delay_spin.value()
        config.runtime.remote_echo_guard_resume_delay_ms = self.runtime_remote_echo_guard_resume_delay_spin.value()
        config.runtime.translation_exact_cache_size = self.runtime_translation_cache_spin.value()
        config.runtime.translation_prefix_min_delta_chars = self.runtime_prefix_delta_spin.value()
        config.runtime.tts_drop_backlog_threshold = self.runtime_tts_drop_threshold_spin.value()

    def set_runtime_status(self, text: str) -> None:
        self.runtime_status_label.setText(text)

    def _build_asr_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.asr_engine_combo = QComboBox()
        self.asr_engine_combo.addItem("faster-whisper", "faster_whisper")
        self._configure_combo_popup(self.asr_engine_combo)

        self.asr_model_combo = QComboBox()
        # 中文 ASR 模型
        self.asr_model_combo.setEditable(True)
        for model in self._discover_asr_models():
            self.asr_model_combo.addItem(model)
        self._configure_combo_popup(self.asr_model_combo)
        self.asr_model_combo.setToolTip("可手動輸入自定義模型名稱或本地路徑")

        self.remote_asr_model_combo = QComboBox()
        # 英文 ASR 模型
        self.remote_asr_model_combo.setEditable(True)
        for model in self._discover_asr_models():
            self.remote_asr_model_combo.addItem(model)
        self._configure_combo_popup(self.remote_asr_model_combo)
        self.remote_asr_model_combo.setToolTip("可手動輸入自定義模型名稱或本地路徑")

        self.asr_device_combo = QComboBox()
        self.asr_device_combo.addItem("cuda", "cuda")
        self.asr_device_combo.addItem("cpu", "cpu")
        self._configure_combo_popup(self.asr_device_combo)
        self.asr_device_combo.setMinimumWidth(150)
        self.asr_device_combo.setMaximumWidth(150)

        self.asr_compute_label = QLabel("-")
        self.asr_compute_label.setMinimumWidth(78)
        self.asr_compute_label.setMaximumWidth(78)

        self.asr_beam_spin = QSpinBox()
        self.asr_beam_spin.setRange(1, 8)
        self.remote_asr_beam_spin = QSpinBox()
        self.remote_asr_beam_spin.setRange(1, 8)
        self.asr_condition_prev_check = QCheckBox("延續前文")
        self.remote_asr_condition_prev_check = QCheckBox("延續前文")
        self.asr_partial_interval_spin = QSpinBox()
        self.asr_partial_interval_spin.setRange(200, 5000)
        self.remote_asr_partial_interval_spin = QSpinBox()
        self.remote_asr_partial_interval_spin.setRange(200, 5000)
        self.asr_partial_history_spin = QSpinBox()
        self.asr_partial_history_spin.setRange(1, 30)
        self.remote_asr_partial_history_spin = QSpinBox()
        self.remote_asr_partial_history_spin.setRange(1, 30)
        self.asr_final_history_spin = QSpinBox()
        self.asr_final_history_spin.setRange(1, 60)
        self.remote_asr_final_history_spin = QSpinBox()
        self.remote_asr_final_history_spin.setRange(1, 60)
        self.asr_vad_enabled = QCheckBox("VAD")
        self.remote_asr_vad_enabled = QCheckBox("VAD")
        self.asr_min_speech_spin = QSpinBox()
        self.asr_min_speech_spin.setRange(100, 4000)
        self.remote_asr_min_speech_spin = QSpinBox()
        self.remote_asr_min_speech_spin.setRange(100, 4000)
        self.asr_min_silence_spin = QSpinBox()
        self.asr_min_silence_spin.setRange(300, 8000)
        self.remote_asr_min_silence_spin = QSpinBox()
        self.remote_asr_min_silence_spin.setRange(300, 8000)
        self.asr_speech_pad_spin = QSpinBox()
        self.asr_speech_pad_spin.setRange(0, 2000)
        self.remote_asr_speech_pad_spin = QSpinBox()
        self.remote_asr_speech_pad_spin.setRange(0, 2000)
        self.asr_max_speech_spin = QSpinBox()
        self.asr_max_speech_spin.setRange(3, 60)
        self.remote_asr_max_speech_spin = QSpinBox()
        self.remote_asr_max_speech_spin.setRange(3, 60)
        self.asr_rms_threshold_spin = QDoubleSpinBox()
        self.asr_rms_threshold_spin.setRange(0.001, 0.2)
        self.asr_rms_threshold_spin.setSingleStep(0.001)
        self.asr_rms_threshold_spin.setDecimals(3)
        self.remote_asr_rms_threshold_spin = QDoubleSpinBox()
        self.remote_asr_rms_threshold_spin.setRange(0.001, 0.2)
        self.remote_asr_rms_threshold_spin.setSingleStep(0.001)
        self.remote_asr_rms_threshold_spin.setDecimals(3)
        self.asr_no_speech_threshold_spin = QDoubleSpinBox()
        self.asr_no_speech_threshold_spin.setRange(0.0, 1.0)
        self.asr_no_speech_threshold_spin.setSingleStep(0.05)
        self.asr_no_speech_threshold_spin.setDecimals(2)
        self.remote_asr_no_speech_threshold_spin = QDoubleSpinBox()
        self.remote_asr_no_speech_threshold_spin.setRange(0.0, 1.0)
        self.remote_asr_no_speech_threshold_spin.setSingleStep(0.05)
        self.remote_asr_no_speech_threshold_spin.setDecimals(2)
        self.asr_temperature_fallback_local_edit = QLineEdit()
        self.asr_temperature_fallback_local_edit.setPlaceholderText("0.0,0.2")
        self.asr_temperature_fallback_remote_edit = QLineEdit()
        self.asr_temperature_fallback_remote_edit.setPlaceholderText("0.0,0.2,0.4")
        self.asr_queue_local_spin = QSpinBox()
        self.asr_queue_local_spin.setRange(4, 512)
        self.asr_queue_remote_spin = QSpinBox()
        self.asr_queue_remote_spin.setRange(4, 512)
        self._set_dual_field_style(
            self.asr_model_combo,
            self.remote_asr_model_combo,
            self.asr_beam_spin,
            self.remote_asr_beam_spin,
            self.asr_partial_interval_spin,
            self.remote_asr_partial_interval_spin,
            self.asr_partial_history_spin,
            self.remote_asr_partial_history_spin,
            self.asr_final_history_spin,
            self.remote_asr_final_history_spin,
            self.asr_min_speech_spin,
            self.remote_asr_min_speech_spin,
            self.asr_min_silence_spin,
            self.remote_asr_min_silence_spin,
            self.asr_speech_pad_spin,
            self.remote_asr_speech_pad_spin,
            self.asr_max_speech_spin,
            self.remote_asr_max_speech_spin,
            self.asr_rms_threshold_spin,
            self.remote_asr_rms_threshold_spin,
            self.asr_no_speech_threshold_spin,
            self.remote_asr_no_speech_threshold_spin,
            self.asr_queue_local_spin,
            self.asr_queue_remote_spin,
            self.asr_condition_prev_check,
            self.remote_asr_condition_prev_check,
            self.asr_vad_enabled,
            self.remote_asr_vad_enabled,
            self.asr_temperature_fallback_local_edit,
            self.asr_temperature_fallback_remote_edit,
        )

        asr_device_row = self._build_inline_row(self.asr_device_combo, self.asr_compute_label)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("", self._build_dual_header_row("中文", "英文"))
        form.addRow("模型", self._build_dual_field_row(self.asr_model_combo, self.remote_asr_model_combo))
        form.addRow("引擎(共用)", self.asr_engine_combo)
        form.addRow("裝置/精度(共用)", asr_device_row)
        form.addRow("溫度 fallback", self._build_dual_field_row(self.asr_temperature_fallback_local_edit, self.asr_temperature_fallback_remote_edit))
        form.addRow("ASR 佇列", self._build_dual_field_row(self.asr_queue_local_spin, self.asr_queue_remote_spin))
        form.addRow("Beam 寬度", self._build_dual_field_row(self.asr_beam_spin, self.remote_asr_beam_spin))
        zh_cond_vad = self._build_inline_row(self.asr_condition_prev_check, self.asr_vad_enabled)
        en_cond_vad = self._build_inline_row(self.remote_asr_condition_prev_check, self.remote_asr_vad_enabled)
        form.addRow("延續前文/VAD", self._build_dual_field_row(zh_cond_vad, en_cond_vad))
        form.addRow(
            "局部更新間隔(ms)",
            self._build_dual_field_row(self.asr_partial_interval_spin, self.remote_asr_partial_interval_spin),
        )
        form.addRow("Partial 保留(s)", self._build_dual_field_row(self.asr_partial_history_spin, self.remote_asr_partial_history_spin))
        form.addRow("Final 保留(s)", self._build_dual_field_row(self.asr_final_history_spin, self.remote_asr_final_history_spin))
        form.addRow("最短語音(ms)", self._build_dual_field_row(self.asr_min_speech_spin, self.remote_asr_min_speech_spin))
        form.addRow("最短靜音(ms)", self._build_dual_field_row(self.asr_min_silence_spin, self.remote_asr_min_silence_spin))
        form.addRow("語音補白(ms)", self._build_dual_field_row(self.asr_speech_pad_spin, self.remote_asr_speech_pad_spin))
        form.addRow("最長語音(s)", self._build_dual_field_row(self.asr_max_speech_spin, self.remote_asr_max_speech_spin))
        form.addRow("No Speech 門檻", self._build_dual_field_row(self.asr_no_speech_threshold_spin, self.remote_asr_no_speech_threshold_spin))
        form.addRow("RMS 門檻", self._build_dual_field_row(self.asr_rms_threshold_spin, self.remote_asr_rms_threshold_spin))

        group = QGroupBox("語音辨識 ASR")
        group.setLayout(form)
        return group, form

    def _build_llm_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.llm_backend_combo = QComboBox()
        self.llm_backend_combo.addItem("LM Studio", "lm_studio")
        self._configure_combo_popup(self.llm_backend_combo)
        self.llm_url_label = QLabel("http://127.0.0.1:1234")
        self.llm_url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.llm_model_combo = QComboBox()
        # 中翻英模型
        self.llm_model_combo.setEditable(False)
        self._configure_combo_popup(self.llm_model_combo)
        self.remote_llm_model_combo = QComboBox()
        # 英翻中模型
        self.remote_llm_model_combo.setEditable(False)
        self._configure_combo_popup(self.remote_llm_model_combo)
        self.llm_reload_models_btn = QPushButton("重新載入模型")
        self.llm_timeout_spin = QSpinBox()
        self.llm_timeout_spin.setRange(5, 120)
        self.remote_llm_timeout_spin = QSpinBox()
        self.remote_llm_timeout_spin.setRange(5, 120)
        self.llm_temperature_spin = QDoubleSpinBox()
        self.llm_temperature_spin.setRange(0.0, 2.0)
        self.llm_temperature_spin.setSingleStep(0.1)
        self.llm_temperature_spin.setDecimals(2)
        self.remote_llm_temperature_spin = QDoubleSpinBox()
        self.remote_llm_temperature_spin.setRange(0.0, 2.0)
        self.remote_llm_temperature_spin.setSingleStep(0.1)
        self.remote_llm_temperature_spin.setDecimals(2)
        self.llm_top_p_spin = QDoubleSpinBox()
        self.llm_top_p_spin.setRange(0.0, 1.0)
        self.llm_top_p_spin.setSingleStep(0.05)
        self.llm_top_p_spin.setDecimals(2)
        self.remote_llm_top_p_spin = QDoubleSpinBox()
        self.remote_llm_top_p_spin.setRange(0.0, 1.0)
        self.remote_llm_top_p_spin.setSingleStep(0.05)
        self.remote_llm_top_p_spin.setDecimals(2)
        self.llm_max_tokens_spin = QSpinBox()
        self.llm_max_tokens_spin.setRange(16, 2048)
        self.remote_llm_max_tokens_spin = QSpinBox()
        self.remote_llm_max_tokens_spin.setRange(16, 2048)
        self.llm_repeat_penalty_spin = QDoubleSpinBox()
        self.llm_repeat_penalty_spin.setRange(0.8, 2.0)
        self.llm_repeat_penalty_spin.setSingleStep(0.01)
        self.llm_repeat_penalty_spin.setDecimals(2)
        self.remote_llm_repeat_penalty_spin = QDoubleSpinBox()
        self.remote_llm_repeat_penalty_spin.setRange(0.8, 2.0)
        self.remote_llm_repeat_penalty_spin.setSingleStep(0.01)
        self.remote_llm_repeat_penalty_spin.setDecimals(2)
        self.llm_stop_tokens_edit = QPlainTextEdit()
        self.llm_stop_tokens_edit.setPlaceholderText("每行一個 stop token，例如:\n</target>\nTranslation:")
        self.remote_llm_stop_tokens_edit = QPlainTextEdit()
        self.remote_llm_stop_tokens_edit.setPlaceholderText("每行一個 stop token，例如:\n</target>\nTranslation:")
        self._set_multiline_field_style(self.llm_stop_tokens_edit)
        self._set_multiline_field_style(self.remote_llm_stop_tokens_edit)
        self.llm_sliding_window_enabled = QCheckBox("啟用上下文拼接")
        self.remote_llm_sliding_window_enabled = QCheckBox("啟用上下文拼接")
        self.llm_trigger_tokens_spin = QSpinBox()
        self.llm_trigger_tokens_spin.setRange(5, 120)
        self.remote_llm_trigger_tokens_spin = QSpinBox()
        self.remote_llm_trigger_tokens_spin.setRange(5, 120)
        self.llm_context_items_spin = QSpinBox()
        self.llm_context_items_spin.setRange(2, 20)
        self.remote_llm_context_items_spin = QSpinBox()
        self.remote_llm_context_items_spin.setRange(2, 20)
        self.llm_queue_local_spin = QSpinBox()
        self.llm_queue_local_spin.setRange(4, 512)
        self.llm_queue_remote_spin = QSpinBox()
        self.llm_queue_remote_spin.setRange(4, 512)
        self._set_dual_field_style(
            self.llm_model_combo,
            self.remote_llm_model_combo,
            self.llm_timeout_spin,
            self.remote_llm_timeout_spin,
            self.llm_temperature_spin,
            self.remote_llm_temperature_spin,
            self.llm_top_p_spin,
            self.remote_llm_top_p_spin,
            self.llm_max_tokens_spin,
            self.remote_llm_max_tokens_spin,
            self.llm_repeat_penalty_spin,
            self.remote_llm_repeat_penalty_spin,
            self.llm_trigger_tokens_spin,
            self.remote_llm_trigger_tokens_spin,
            self.llm_context_items_spin,
            self.remote_llm_context_items_spin,
            self.llm_queue_local_spin,
            self.llm_queue_remote_spin,
            self.llm_sliding_window_enabled,
            self.remote_llm_sliding_window_enabled,
        )
        llm_service_row = self._build_inline_row(self.llm_url_label, self.llm_reload_models_btn)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("", self._build_dual_header_row("中翻英", "英翻中"))
        form.addRow("模型", self._build_dual_field_row(self.llm_model_combo, self.remote_llm_model_combo))
        form.addRow("後端(共用)", self.llm_backend_combo)
        form.addRow("服務位址(共用)", llm_service_row)
        form.addRow("LLM 佇列", self._build_dual_field_row(self.llm_queue_local_spin, self.llm_queue_remote_spin))
        form.addRow("逾時(秒)", self._build_dual_field_row(self.llm_timeout_spin, self.remote_llm_timeout_spin))
        form.addRow("溫度", self._build_dual_field_row(self.llm_temperature_spin, self.remote_llm_temperature_spin))
        form.addRow("Top P", self._build_dual_field_row(self.llm_top_p_spin, self.remote_llm_top_p_spin))
        form.addRow("最大輸出 Token", self._build_dual_field_row(self.llm_max_tokens_spin, self.remote_llm_max_tokens_spin))
        form.addRow("Repeat Penalty", self._build_dual_field_row(self.llm_repeat_penalty_spin, self.remote_llm_repeat_penalty_spin))
        form.addRow("Stop Tokens", self._build_dual_field_row(self.llm_stop_tokens_edit, self.remote_llm_stop_tokens_edit))
        form.addRow("啟用上下文拼接", self._build_dual_field_row(self.llm_sliding_window_enabled, self.remote_llm_sliding_window_enabled))
        form.addRow("觸發 Token", self._build_dual_field_row(self.llm_trigger_tokens_spin, self.remote_llm_trigger_tokens_spin))
        form.addRow("上下文項數", self._build_dual_field_row(self.llm_context_items_spin, self.remote_llm_context_items_spin))

        group = QGroupBox("翻譯模型 LLM")
        group.setLayout(form)
        return group, form

    def _build_runtime_group(self) -> tuple[QGroupBox, QFormLayout]:
        self.runtime_sample_rate_spin = QComboBox()
        self.runtime_sample_rate_spin.setEditable(False)
        for r in _COMMON_SAMPLE_RATES:
            self.runtime_sample_rate_spin.addItem(str(r), r)
        self._configure_combo_popup(self.runtime_sample_rate_spin)
        self.runtime_chunk_spin = QSpinBox()
        self.runtime_chunk_spin.setRange(20, 1000)
        self.runtime_asr_pre_roll_spin = QSpinBox()
        self.runtime_asr_pre_roll_spin.setRange(0, 2000)
        self.runtime_tts_cancel_pending_check = QCheckBox("新句取消舊語音")
        self.runtime_tts_cancel_policy_combo = QComboBox()
        self.runtime_tts_cancel_policy_combo.addItem("取消所有未播", "all_pending")
        self.runtime_tts_cancel_policy_combo.addItem("只取消舊句", "older_only")
        self._configure_combo_popup(self.runtime_tts_cancel_policy_combo)
        self.runtime_tts_max_wait_spin = QSpinBox()
        self.runtime_tts_max_wait_spin.setRange(500, 15000)
        self.runtime_tts_max_chars_spin = QSpinBox()
        self.runtime_tts_max_chars_spin.setRange(20, 2000)
        self.runtime_tts_drop_threshold_spin = QSpinBox()
        self.runtime_tts_drop_threshold_spin.setRange(2, 20)
        self.runtime_translation_cache_spin = QSpinBox()
        self.runtime_translation_cache_spin.setRange(64, 4096)
        self.runtime_prefix_delta_spin = QSpinBox()
        self.runtime_prefix_delta_spin.setRange(1, 50)
        self.runtime_llm_streaming_tokens_spin = QSpinBox()
        self.runtime_llm_streaming_tokens_spin.setRange(8, 256)
        self.runtime_max_pipeline_latency_spin = QSpinBox()
        self.runtime_max_pipeline_latency_spin.setRange(500, 20000)
        self.runtime_local_echo_guard_enabled_check = QCheckBox("啟用本地防回授")
        self.runtime_local_echo_guard_resume_delay_spin = QSpinBox()
        self.runtime_local_echo_guard_resume_delay_spin.setRange(0, 5000)
        self.runtime_remote_echo_guard_resume_delay_spin = QSpinBox()
        self.runtime_remote_echo_guard_resume_delay_spin.setRange(0, 5000)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("執行取樣率", self.runtime_sample_rate_spin)
        form.addRow("音訊切片(ms)", self.runtime_chunk_spin)
        form.addRow("ASR Pre-roll(ms)", self.runtime_asr_pre_roll_spin)
        form.addRow("翻譯快取大小", self.runtime_translation_cache_spin)
        form.addRow("局部觸發差異字數", self.runtime_prefix_delta_spin)
        form.addRow("Streaming tokens", self.runtime_llm_streaming_tokens_spin)
        form.addRow("最大 pipeline 延遲(ms)", self.runtime_max_pipeline_latency_spin)
        form.addRow("本地防回授恢復(ms)", self.runtime_local_echo_guard_resume_delay_spin)
        form.addRow("遠端防回授恢復(ms)", self.runtime_remote_echo_guard_resume_delay_spin)

        group = QGroupBox("系統執行")
        group.setLayout(form)
        return group, form

    def _create_tts_widgets(self, title: str) -> TtsWidgets:
        engine_combo = QComboBox()
        engine_combo.addItem("Edge TTS（雲端）", "edge_tts")
        self._configure_combo_popup(engine_combo)

        edge_voice_combo = QComboBox()
        for label, voice_name in _EDGE_VOICE_OPTIONS:
            edge_voice_combo.addItem(label, voice_name)
        self._configure_combo_popup(edge_voice_combo)

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
        sample_rate_spin = QComboBox()
        sample_rate_spin.setEditable(True)
        for r in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
            sample_rate_spin.addItem(str(r), r)
        self._configure_combo_popup(sample_rate_spin)

        for editor in (exec_edit, model_edit, config_edit):
            editor.setReadOnly(True)
            editor.setCursor(Qt.CursorShape.PointingHandCursor)
            editor.setToolTip("點一下選擇檔案")
            self._set_uniform_field_style(editor, minimum_width=220)

        for compact in (
            engine_combo,
            edge_voice_combo,
            speaker_combo,
            length_scale_spin,
            noise_scale_spin,
            noise_w_spin,
            sample_rate_spin,
        ):
            self._set_uniform_field_style(compact, minimum_width=220)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("語速 Length Scale", length_scale_spin)
        form.addRow("隨機度 Noise Scale", noise_scale_spin)
        form.addRow("音色 Noise W", noise_w_spin)
        form.addRow("取樣率", sample_rate_spin)
        form.addRow("TTS 丟棄積壓門檻", self.runtime_tts_drop_threshold_spin)
        form.addRow("TTS 最大等待(ms)", self.runtime_tts_max_wait_spin)
        form.addRow("TTS 最大字數", self.runtime_tts_max_chars_spin)
        form.addRow("取消策略", self.runtime_tts_cancel_policy_combo)
        cancel_and_guard = self._build_inline_row(self.runtime_tts_cancel_pending_check, self.runtime_local_echo_guard_enabled_check)
        form.addRow("", cancel_and_guard)

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

    def _create_tts_channel_widgets(self, title: str, language_kind: str) -> TtsChannelWidgets:
        engine_combo = QComboBox()
        engine_combo.addItem("Edge TTS（雲端）", "edge_tts")
        self._configure_combo_popup(engine_combo)

        edge_voice_combo = QComboBox()
        for label, voice_name in self._edge_voice_options_for(language_kind):
            edge_voice_combo.addItem(label, voice_name)
        self._configure_combo_popup(edge_voice_combo)

        sample_rate_spin = QComboBox()
        sample_rate_spin.setEditable(False)
        for r in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
            sample_rate_spin.addItem(str(r), r)
        self._configure_combo_popup(sample_rate_spin)

        noise_w_spin = QDoubleSpinBox()
        noise_w_spin.setRange(0.0, 2.0)
        noise_w_spin.setSingleStep(0.05)
        noise_w_spin.setDecimals(3)
        noise_w_spin.setValue(0.6)

        for compact in (engine_combo, edge_voice_combo, sample_rate_spin, noise_w_spin):
            self._set_uniform_field_style(compact, minimum_width=220)

        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("引擎", engine_combo)
        form.addRow("Edge 聲線", edge_voice_combo)
        form.addRow("取樣率", sample_rate_spin)
        form.addRow("Noise W", noise_w_spin)

        group = QGroupBox(title)
        group.setLayout(form)
        return TtsChannelWidgets(
            group=group,
            form=form,
            engine_combo=engine_combo,
            edge_voice_combo=edge_voice_combo,
            sample_rate_spin=sample_rate_spin,
            noise_w_spin=noise_w_spin,
        )

    def _build_tts_channel_compact_group(self) -> QGroupBox:
        group = QGroupBox("輸出通道覆寫（共用欄位）")
        form = QFormLayout()
        self._configure_form_layout(form)
        form.addRow("", self._build_dual_header_row("中文", "英文"))
        form.addRow(
            "引擎",
            self._build_dual_field_row(
                self.local_tts_override.engine_combo,
                self.remote_tts_override.engine_combo,
            ),
        )
        form.addRow(
            "Edge 聲線",
            self._build_dual_field_row(
                self.local_tts_override.edge_voice_combo,
                self.remote_tts_override.edge_voice_combo,
            ),
        )
        form.addRow(
            "取樣率",
            self._build_dual_field_row(
                self.local_tts_override.sample_rate_spin,
                self.remote_tts_override.sample_rate_spin,
            ),
        )
        form.addRow(
            "Noise W",
            self._build_dual_field_row(
                self.local_tts_override.noise_w_spin,
                self.remote_tts_override.noise_w_spin,
            ),
        )
        self.tts_queue_local_spin = QSpinBox()
        self.tts_queue_local_spin.setRange(4, 512)
        self.tts_queue_remote_spin = QSpinBox()
        self.tts_queue_remote_spin.setRange(4, 512)
        self._set_dual_field_style(self.tts_queue_local_spin, self.tts_queue_remote_spin)
        form.addRow("TTS 佇列", self._build_dual_field_row(self.tts_queue_local_spin, self.tts_queue_remote_spin))
        group.setLayout(form)
        return group

    @staticmethod
    def _edge_voice_options_for(language_kind: str) -> list[tuple[str, str]]:
        normalized = (language_kind or "").strip().lower()
        if normalized.startswith("en"):
            return [item for item in _EDGE_VOICE_OPTIONS if item[1].lower().startswith("en-")]
        return [item for item in _EDGE_VOICE_OPTIONS if item[1].lower().startswith("zh-")]

    def _wire_events(self) -> None:
        for compact in (
            self.asr_engine_combo,
            self.llm_backend_combo,
            self.runtime_sample_rate_spin,
            self.runtime_chunk_spin,
            self.runtime_translation_cache_spin,
            self.runtime_prefix_delta_spin,
            self.runtime_tts_drop_threshold_spin,
            self.runtime_tts_cancel_pending_check,
            self.runtime_tts_cancel_policy_combo,
            self.runtime_tts_max_wait_spin,
            self.runtime_tts_max_chars_spin,
            self.runtime_llm_streaming_tokens_spin,
            self.runtime_max_pipeline_latency_spin,
            self.runtime_local_echo_guard_enabled_check,
            self.runtime_local_echo_guard_resume_delay_spin,
            self.runtime_remote_echo_guard_resume_delay_spin,
        ):
            self._set_uniform_field_style(compact, minimum_width=140)

        self._set_uniform_field_style(self.asr_device_combo, minimum_width=150)

        self.llm_url_label.setMinimumWidth(140)
        self.llm_url_label.setMinimumHeight(_CONTROL_HEIGHT)
        self.asr_compute_label.setMinimumHeight(_CONTROL_HEIGHT)
        self.llm_reload_models_btn.setMinimumHeight(_CONTROL_HEIGHT)
        self.llm_backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.asr_device_combo.currentIndexChanged.connect(self._on_asr_device_changed)
        self.llm_reload_models_btn.clicked.connect(self._reload_llm_models)

        for widget in (
            self.asr_engine_combo,
            self.asr_model_combo,
            self.remote_asr_model_combo,
            self.asr_device_combo,
            self.asr_beam_spin,
            self.remote_asr_beam_spin,
            self.asr_partial_interval_spin,
            self.remote_asr_partial_interval_spin,
            self.asr_partial_history_spin,
            self.remote_asr_partial_history_spin,
            self.asr_final_history_spin,
            self.remote_asr_final_history_spin,
            self.asr_vad_enabled,
            self.remote_asr_vad_enabled,
            self.asr_condition_prev_check,
            self.remote_asr_condition_prev_check,
            self.asr_temperature_fallback_local_edit,
            self.asr_temperature_fallback_remote_edit,
            self.asr_min_speech_spin,
            self.remote_asr_min_speech_spin,
            self.asr_min_silence_spin,
            self.remote_asr_min_silence_spin,
            self.asr_speech_pad_spin,
            self.remote_asr_speech_pad_spin,
            self.asr_max_speech_spin,
            self.remote_asr_max_speech_spin,
            self.asr_no_speech_threshold_spin,
            self.remote_asr_no_speech_threshold_spin,
            self.asr_rms_threshold_spin,
            self.remote_asr_rms_threshold_spin,
            self.llm_backend_combo,
            self.llm_model_combo,
            self.remote_llm_model_combo,
            self.llm_timeout_spin,
            self.remote_llm_timeout_spin,
            self.llm_temperature_spin,
            self.remote_llm_temperature_spin,
            self.llm_top_p_spin,
            self.remote_llm_top_p_spin,
            self.llm_max_tokens_spin,
            self.remote_llm_max_tokens_spin,
            self.llm_repeat_penalty_spin,
            self.remote_llm_repeat_penalty_spin,
            self.llm_stop_tokens_edit,
            self.remote_llm_stop_tokens_edit,
            self.llm_sliding_window_enabled,
            self.remote_llm_sliding_window_enabled,
            self.llm_trigger_tokens_spin,
            self.remote_llm_trigger_tokens_spin,
            self.llm_context_items_spin,
            self.remote_llm_context_items_spin,
            self.runtime_sample_rate_spin,
            self.runtime_chunk_spin,
            self.runtime_asr_pre_roll_spin,
            self.asr_queue_local_spin,
            self.asr_queue_remote_spin,
            self.llm_queue_local_spin,
            self.llm_queue_remote_spin,
            self.tts_queue_local_spin,
            self.tts_queue_remote_spin,
            self.runtime_translation_cache_spin,
            self.runtime_prefix_delta_spin,
            self.runtime_tts_drop_threshold_spin,
            self.runtime_tts_cancel_pending_check,
            self.runtime_tts_cancel_policy_combo,
            self.runtime_tts_max_wait_spin,
            self.runtime_tts_max_chars_spin,
            self.runtime_llm_streaming_tokens_spin,
            self.runtime_max_pipeline_latency_spin,
            self.runtime_local_echo_guard_enabled_check,
            self.runtime_local_echo_guard_resume_delay_spin,
            self.runtime_remote_echo_guard_resume_delay_spin,
        ):
            self._connect_change_signal(widget)

        self._bind_tts_widget_events(self.base_tts, "主設定")
        self._bind_tts_channel_events(self.local_tts_override)
        self._bind_tts_channel_events(self.remote_tts_override)

    def _bind_tts_widget_events(self, widgets: TtsWidgets, title: str) -> None:
        widgets.engine_combo.setVisible(False)
        widgets.edge_voice_combo.setVisible(False)
        widgets.exec_edit.setVisible(False)
        widgets.model_edit.setVisible(False)
        widgets.config_edit.setVisible(False)
        widgets.speaker_combo.setVisible(False)
        for widget in (
            widgets.length_scale_spin,
            widgets.noise_scale_spin,
            widgets.noise_w_spin,
            widgets.sample_rate_spin,
        ):
            self._connect_change_signal(widget)

    def _bind_tts_channel_events(self, widgets: TtsChannelWidgets) -> None:
        for widget in (
            widgets.engine_combo,
            widgets.edge_voice_combo,
            widgets.sample_rate_spin,
            widgets.noise_w_spin,
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
        self._select_combo_data(widgets.sample_rate_spin, config.sample_rate)
        self._reload_tts_speaker_options(widgets, selected_id=config.speaker_id)

    @staticmethod
    def _update_tts_config(widgets: TtsWidgets, config: TtsConfig) -> None:
        config.engine = "edge_tts"
        # Base profile only stores shared synthesis params; per-channel voice is set via overrides.
        if not (config.voice_name or "").strip():
            config.voice_name = "zh-TW-HsiaoChenNeural"
        config.executable_path = ""
        config.model_path = ""
        config.config_path = ""
        config.speaker_id = 0
        config.length_scale = float(widgets.length_scale_spin.value())
        config.noise_scale = float(widgets.noise_scale_spin.value())
        config.noise_w = float(widgets.noise_w_spin.value())
        try:
            config.sample_rate = int(widgets.sample_rate_spin.currentData() or 0)
            if config.sample_rate == 0:
                config.sample_rate = int(str(widgets.sample_rate_spin.currentText()).strip() or 0)
        except Exception:
            config.sample_rate = int(str(widgets.sample_rate_spin.currentText()).strip() or 0)

    def _apply_tts_override(self, widgets: TtsChannelWidgets, override: TtsChannelOverride, *, fallback: TtsConfig) -> None:
        self._select_combo_data(widgets.engine_combo, override.engine or "")
        voice = override.voice_name if override.voice_name is not None else fallback.voice_name
        self._select_combo_data(widgets.edge_voice_combo, voice or "")
        self._select_combo_data(widgets.sample_rate_spin, int(override.sample_rate) if override.sample_rate is not None else 0)
        widgets.noise_w_spin.setValue(float(override.noise_w) if override.noise_w is not None else 0.0)

    @staticmethod
    def _update_tts_override(widgets: TtsChannelWidgets) -> TtsChannelOverride:
        engine = str(widgets.engine_combo.currentData() or "")
        voice_name = str(widgets.edge_voice_combo.currentData() or "")
        try:
            sample_rate = int(widgets.sample_rate_spin.currentData() or 0)
            if sample_rate == 0:
                sample_rate = int(str(widgets.sample_rate_spin.currentText()).strip() or 0)
        except Exception:
            sample_rate = int(str(widgets.sample_rate_spin.currentText()).strip() or 0)
        noise_w = widgets.noise_w_spin.value()
        return TtsChannelOverride(
            engine=engine or None,
            voice_name=voice_name or None,
            sample_rate=sample_rate if sample_rate > 0 else None,
            noise_w=noise_w if noise_w > 0 else None,
        )

    def _reload_tts_speaker_options(self, widgets: TtsWidgets, selected_id: int | None = None) -> None:
        speaker_rows = self._load_speaker_rows_from_tts_config(widgets.config_edit.text().strip())
        if not speaker_rows:
            speaker_rows = [("預設聲線", 0)]

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
        self._configure_combo_popup(widgets.speaker_combo)

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
                    rows.append((f"聲線：{name}", int(sid)))
                except Exception:
                    continue
            rows.sort(key=lambda row: row[1])
            return rows

        speakers = data.get("speakers")
        if isinstance(speakers, list):
            rows = []
            for idx, item in enumerate(speakers):
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("speaker") or f"聲線 {idx + 1}")
                else:
                    name = str(item)
                rows.append((f"聲線：{name}", idx))
            return rows

        return []

    def _reload_llm_models(self) -> None:
        if self._model_loading:
            return
        backend = str(self.llm_backend_combo.currentData() or "lm_studio")
        url = self._backend_url(backend)
        self._model_loading = True
        self.llm_reload_models_btn.setEnabled(False)
        self.llm_reload_models_btn.setText("載入中...")

        def _worker() -> None:
            try:
                client = LmStudioClient(base_url=url, model="", request_timeout_sec=3.0)
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
        self.llm_reload_models_btn.setText("重新載入模型")

        if not ok:
            self.runtime_status_label.setText(f"LLM 模型載入失敗：{payload}")
            return

        current = (self._preferred_llm_model or self.llm_model_combo.currentText().strip()).strip()
        remote_current_before = self.remote_llm_model_combo.currentText().strip()
        models = payload if isinstance(payload, list) else []
        if not models:
            self.runtime_status_label.setText("找不到可用的 LLM 模型")
            return

        if current and current not in models:
            models = [current, *models]

        self.llm_model_combo.blockSignals(True)
        self.remote_llm_model_combo.blockSignals(True)
        self.llm_model_combo.clear()
        self.remote_llm_model_combo.clear()
        for model in models:
            self.llm_model_combo.addItem(model)
            self.remote_llm_model_combo.addItem(model)
        self.llm_model_combo.blockSignals(False)
        self.remote_llm_model_combo.blockSignals(False)
        self._configure_combo_popup(self.llm_model_combo)
        self._configure_combo_popup(self.remote_llm_model_combo)

        if current:
            self._set_combo_text(self.llm_model_combo, current)
        else:
            self._set_combo_text(self.llm_model_combo, models[0])
        remote_current = remote_current_before or models[0]
        if remote_current:
            self._set_combo_text(self.remote_llm_model_combo, remote_current)
        self._preferred_llm_model = self.llm_model_combo.currentText().strip()

    def _on_backend_changed(self) -> None:
        backend = str(self.llm_backend_combo.currentData() or "lm_studio")
        self._apply_backend_url(backend, self.llm_url_label)
        self._reload_llm_models()
        self._notify_settings_changed()

    def _on_asr_device_changed(self) -> None:
        device = str(self.asr_device_combo.currentData() or "cuda")
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
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(_FORM_V_SPACING)

    @staticmethod
    def _build_inline_row(*widgets: QWidget) -> QWidget:
        container = QWidget()
        container.setMinimumHeight(_CONTROL_HEIGHT)
        container.setMaximumHeight(_CONTROL_HEIGHT)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        for widget in widgets:
            row.addWidget(widget)
        row.addStretch(1)
        return container

    @staticmethod
    def _build_dual_header_row(left_title: str, right_title: str) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        left = QLabel(left_title)
        right = QLabel(right_title)
        left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(left, 1)
        row.addWidget(right, 1)
        return container

    @staticmethod
    def _build_dual_field_row(left_widget: QWidget, right_widget: QWidget) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(left_widget, 1)
        row.addSpacing(8)
        row.addWidget(right_widget, 1)
        return container

    @staticmethod
    def _set_uniform_field_style(widget: QWidget, minimum_width: int = 220, control_height: int = _CONTROL_HEIGHT) -> None:
        widget.setMinimumWidth(minimum_width)
        widget.setMinimumHeight(control_height)
        widget.setMaximumHeight(control_height)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _set_multiline_field_style(widget: QPlainTextEdit, minimum_width: int = 220, control_height: int = 72) -> None:
        widget.setMinimumWidth(minimum_width)
        widget.setMinimumHeight(control_height)
        widget.setMaximumHeight(control_height)
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
            label.setMinimumHeight(_CONTROL_HEIGHT)
            label.setMaximumHeight(_CONTROL_HEIGHT)

    def _apply_group_heights(
        self,
        *,
        asr_form: QFormLayout,
        llm_form: QFormLayout,
        runtime_form: QFormLayout,
        base_tts_form: QFormLayout,
        local_tts_override_form: QFormLayout,
        remote_tts_override_form: QFormLayout,
    ) -> None:
        llm_height = self._estimate_group_height(llm_form)
        runtime_height = self._estimate_group_height(runtime_form)
        asr_height = max(self._estimate_group_height(asr_form), llm_height + runtime_height + _GROUP_ROW_SPACING)
        base_tts_height = self._estimate_group_height(base_tts_form)
        tts_override_height = max(
            self._estimate_group_height(local_tts_override_form),
            self._estimate_group_height(remote_tts_override_form),
        )

        self.asr_group.setMinimumHeight(asr_height)
        self.llm_group.setMinimumHeight(llm_height)
        self.runtime_group.setMinimumHeight(runtime_height)
        self.base_tts.group.setMinimumHeight(base_tts_height)
        self.local_tts_override.group.setMinimumHeight(tts_override_height)
        self.remote_tts_override.group.setMinimumHeight(tts_override_height)
        self.setMinimumHeight(
            asr_height + base_tts_height + tts_override_height + (_GROUP_ROW_SPACING * 3) + _PAGE_MIN_PADDING
        )

    @staticmethod
    def _estimate_group_height(form: QFormLayout) -> int:
        row_count = max(1, form.rowCount())
        return (_CONTROL_HEIGHT * row_count) + (_FORM_V_SPACING * max(0, row_count - 1)) + _GROUP_EXTRA_HEIGHT

    @staticmethod
    def _compute_type_for_device(device: str) -> str:
        if device == "cpu":
            return "int8"
        return "float16"

    @staticmethod
    def _set_dual_field_style(*widgets: QWidget) -> None:
        for widget in widgets:
            widget.setMinimumWidth(80)
            widget.setMinimumHeight(_CONTROL_HEIGHT)
            widget.setMaximumHeight(_CONTROL_HEIGHT)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _apply_backend_url(self, backend: str, label: QLabel) -> None:
        label.setText(self._backend_url(backend))

    @staticmethod
    def _backend_url(backend: str) -> str:
        return "http://127.0.0.1:1234"

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
        models = list(_OFFICIAL_FASTER_WHISPER_MODELS)
        # 掃描 HuggingFace 本地快取中已下載的 faster-whisper 模型
        try:
            from pathlib import Path
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.is_dir():
                for entry in sorted(hf_cache.iterdir()):
                    name = entry.name
                    if name.startswith("models--Systran--faster-whisper-"):
                        short = name[len("models--Systran--faster-whisper-"):]
                        if short and short not in models:
                            models.insert(0, short)
        except Exception:
            pass
        return models

    @staticmethod
    def _configure_combo_popup(combo: QComboBox) -> None:
        combo.setMaxVisibleItems(max(1, combo.count()))
        view = combo.view()
        if view is None:
            return
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
