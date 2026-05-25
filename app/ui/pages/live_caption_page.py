from __future__ import annotations

from datetime import datetime
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.infra.config.schema import AppConfig
from app.infra.asr.profile_selection import asr_profile_family_for_language
from app.infra.tts.voice_policy import default_voice_for_language
from app.ui.pages._live_caption_config import _LiveCaptionConfigMixin, _CHANNEL_DEFAULTS

_STATUS_TEXT = {
    "idle": "未啟動",
    "preparing": "準備中",
    "running": "已啟動",
}

_STATUS_DOT = {
    "idle": "#6b7280",
    "preparing": "#f59e0b",
    "running": "#16a34a",
}

_LANG_LABELS: dict[str, str] = {
    "en": "英文",
    "zh": "中文",
    "zh-tw": "中文",
    "ja": "日文",
    "ko": "韓文",
    "th": "泰文",
}

_ASR_CHOICES: list[tuple[str, str]] = [
    ("none", "無"),
    ("auto", "自動"),
    ("zh-TW", "中文"),
    ("en", "英文"),
    ("ja", "日文"),
    ("ko", "韓文"),
    ("th", "泰文"),
]

_TRANSLATION_TARGET_CHOICES: list[tuple[str, str]] = [
    ("none", "無"),
    ("zh-TW", "中文"),
    ("en", "英文"),
    ("ja", "日文"),
    ("ko", "韓文"),
    ("th", "泰文"),
]

_CHANNELS: tuple[str, str] = ("remote", "local")

_TTS_VOICE_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "zh-TW": [
        ("zh-TW-HsiaoChenNeural", "中文-女（HsiaoChen）"),
        ("zh-TW-HsiaoYuNeural", "中文-女（HsiaoYu）"),
        ("zh-TW-YunJheNeural", "中文-男（YunJhe）"),
    ],
    "en": [
        ("en-US-JennyNeural", "英文（美）-女（Jenny）"),
        ("en-US-GuyNeural", "英文（美）-男（Guy）"),
        ("en-GB-SoniaNeural", "英文（英）-女（Sonia）"),
        ("en-GB-RyanNeural", "英文（英）-男（Ryan）"),
    ],
    "ja": [
        ("ja-JP-NanamiNeural", "日文-女（Nanami）"),
        ("ja-JP-KeitaNeural", "日文-男（Keita）"),
    ],
    "ko": [
        ("ko-KR-SunHiNeural", "韓文-女（SunHi）"),
        ("ko-KR-InJoonNeural", "韓文-男（InJoon）"),
    ],
    "th": [
        ("th-TH-PremwadeeNeural", "泰文-女（Premwadee）"),
        ("th-TH-NiwatNeural", "泰文-男（Niwat）"),
    ],
}
class LiveCaptionPage(_LiveCaptionConfigMixin, QWidget):
    def __init__(
        self,
        on_clear_clicked: Callable[[], None] | None = None,
        on_start_clicked: Callable[[], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
        on_output_mode_changed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_clear_clicked = on_clear_clicked
        self._on_start_clicked = on_start_clicked
        self._on_settings_changed = on_settings_changed
        self._on_output_mode_changed = on_output_mode_changed
        self._suspend_notify = False
        self._direction_controls_enabled = True
        self._detected_asr_language: dict[str, str] = {"local": "", "remote": ""}
        self._configured_asr_models: dict[str, str] = {
            "chinese": "large-v3-turbo",
            "non_chinese": "large-v3-turbo",
        }
        self._last_emitted_output_mode: str = ""
        self.session_mode_combo = QComboBox()
        self.session_mode_combo.addItem("會議字幕", "meeting")
        self.session_mode_combo.addItem("雙向語音", "dialogue")
        self.session_mode_combo.hide()

        self.meeting_mode_btn = QPushButton("會議字幕")
        self.dialogue_mode_btn = QPushButton("雙向語音")
        self.meeting_mode_btn.setObjectName("meeting_mode_button")
        self.dialogue_mode_btn.setObjectName("dialogue_mode_button")
        self._session_mode_button_group = QButtonGroup(self)
        self._session_mode_button_group.setExclusive(True)
        self._session_mode_button_group.addButton(self.meeting_mode_btn)
        self._session_mode_button_group.addButton(self.dialogue_mode_btn)
        for button in (self.meeting_mode_btn, self.dialogue_mode_btn):
            button.setCheckable(True)
            button.setMinimumWidth(104)
            button.setToolTip("切換會議字幕 / 雙向語音模式。執行中需先停止再切換。")
            button.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 12px;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 6px;
                    background: rgba(255,255,255,0.03);
                    color: #dddddd;
                }
                QPushButton:hover {
                    background: rgba(255,255,255,0.06);
                }
                QPushButton:checked {
                    background: #2b78d6;
                    border-color: #1f5ea8;
                    color: white;
                }
                QPushButton:disabled {
                    color: #808080;
                    background: rgba(255,255,255,0.02);
                }
                """
            )
        self.meeting_mode_btn.clicked.connect(lambda _checked=False: self.set_session_mode("meeting", notify=True))
        self.dialogue_mode_btn.clicked.connect(lambda _checked=False: self.set_session_mode("dialogue", notify=True))

        self.meeting_source_combo = QComboBox()
        self.meeting_source_combo.addItem("系統輸入", "system_input")
        self.meeting_source_combo.addItem("系統輸出 loopback", "system_output_loopback")
        self.meeting_device_combo = QComboBox()
        self.meeting_device_combo.setMinimumWidth(240)
        self.meeting_notice_label = QLabel("TTS 已停用，但仍會產生字幕與翻譯記錄")
        self.meeting_notice_label.setStyleSheet("color: #b7bdc6;")
        self.remote_direct_notice_label = QLabel("遠端→本地：直通中，不辨識、不翻譯、不記錄字幕")
        self.local_direct_notice_label = QLabel("本地→遠端：直通中，不辨識、不翻譯、不記錄字幕")
        for notice in (self.remote_direct_notice_label, self.local_direct_notice_label):
            notice.setStyleSheet("color: #f59e0b;")

        self.remote_asr_combo = QComboBox()
        self.local_asr_combo = QComboBox()
        for code, label in _ASR_CHOICES:
            self.remote_asr_combo.addItem(label, code)
            self.local_asr_combo.addItem(label, code)
        for combo in (self.remote_asr_combo, self.local_asr_combo):
            idx = combo.findData("auto")
            if idx >= 0:
                combo.removeItem(idx)
        self.remote_asr_combo.setCurrentIndex(self.remote_asr_combo.findData(_CHANNEL_DEFAULTS["remote"]["asr_default"]))
        self.local_asr_combo.setCurrentIndex(self.local_asr_combo.findData(_CHANNEL_DEFAULTS["local"]["asr_default"]))
        self._configure_combo_popup(self.remote_asr_combo)
        self._configure_combo_popup(self.local_asr_combo)

        self.remote_lang_combo = QComboBox()
        self.local_lang_combo = QComboBox()
        for code, label in _TRANSLATION_TARGET_CHOICES:
            self.remote_lang_combo.addItem(label, code)
            self.local_lang_combo.addItem(label, code)
        self._configure_combo_popup(self.remote_lang_combo)
        self._configure_combo_popup(self.local_lang_combo)

        # Backward-compatible hidden widgets kept for existing tests and callers.
        self.remote_output_mode_combo = QComboBox()
        self.local_output_mode_combo = QComboBox()
        for combo in (self.remote_output_mode_combo, self.local_output_mode_combo):
            combo.addItem("字幕", "subtitle_only")
            combo.addItem("翻譯語音", "translated_tts")
            combo.addItem("直通", "direct_passthrough")
            self._configure_combo_popup(combo)

        self.remote_tts_voice_combo = QComboBox()
        self.local_tts_voice_combo = QComboBox()
        self.remote_tts_voice_combo.setMinimumWidth(220)
        self.local_tts_voice_combo.setMinimumWidth(220)
        self.remote_tts_voice_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.local_tts_voice_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._configure_combo_popup(self.remote_tts_voice_combo)
        self._configure_combo_popup(self.local_tts_voice_combo)

        self.start_btn = QPushButton("開始")
        self.start_btn.clicked.connect(self._handle_start_clicked)
        self.clear_btn = QPushButton("清空字幕")
        self.clear_btn.clicked.connect(self._handle_clear_clicked)
        # 清空/開始按鈕已移到主頁面 tab 長條同列, 在此隱藏避免畫面重覆
        self.start_btn.hide()
        self.clear_btn.hide()

        self.remote_original = QTextEdit()
        self.remote_translated = QTextEdit()
        self.local_original = QTextEdit()
        self.local_translated = QTextEdit()

        self.export_remote_original_btn = QPushButton("匯出")
        self.export_remote_translated_btn = QPushButton("匯出")
        self.export_local_original_btn = QPushButton("匯出")
        self.export_local_translated_btn = QPushButton("匯出")
        for _btn, _editor, _name in (
            (self.export_remote_original_btn, self.remote_original, "remote_original"),
            (self.export_remote_translated_btn, self.remote_translated, "remote_translated"),
            (self.export_local_original_btn, self.local_original, "local_original"),
            (self.export_local_translated_btn, self.local_translated, "local_translated"),
        ):
            _btn.setFixedSize(52, 20)
            _btn.clicked.connect(lambda checked=False, e=_editor, n=_name: self._export_editor(e, n))

        self._editor_committed: dict[QTextEdit, list[str]] = {
            self.remote_original: [],
            self.remote_translated: [],
            self.local_original: [],
            self.local_translated: [],
        }
        self._editor_has_partial: dict[QTextEdit, bool] = {
            self.remote_original: False,
            self.remote_translated: False,
            self.local_original: False,
            self.local_translated: False,
        }

        self.remote_original_label = QLabel("遠端原文")
        self.remote_translated_label = QLabel("遠端翻譯")
        self.local_original_label = QLabel("本地原文")
        self.local_translated_label = QLabel("本地翻譯")
        self.remote_original_status = QLabel()
        self.remote_translated_status = QLabel()
        self.local_original_status = QLabel()
        self.local_translated_status = QLabel()
        self.remote_asr_label = QLabel("ASR語言")
        self.local_asr_label = QLabel("ASR語言")
        self.remote_target_label = QLabel("翻譯目標")
        self.local_target_label = QLabel("翻譯目標")
        self.remote_output_policy_label = QLabel("輸出策略")
        self.local_output_policy_label = QLabel("輸出策略")
        self.remote_tts_voice_label = QLabel("TTS語音")
        self.local_tts_voice_label = QLabel("TTS語音")
        for editor in (
            self.remote_original,
            self.remote_translated,
            self.local_original,
            self.local_translated,
        ):
            editor.setReadOnly(True)
            # 改為等比擴展，同步四個區域特性，避免附加選單導致不一致
            editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            editor.setMinimumSize(0, 0)
            editor.setMaximumSize(16777215, 16777215)

        for combo in (
            self.meeting_source_combo,
            self.meeting_device_combo,
            self.remote_asr_combo,
            self.local_asr_combo,
            self.remote_lang_combo,
            self.local_lang_combo,
            self.remote_output_mode_combo,
            self.local_output_mode_combo,
            self.remote_tts_voice_combo,
            self.local_tts_voice_combo,
        ):
            combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            if combo in (self.remote_tts_voice_combo, self.local_tts_voice_combo):
                combo.setFixedWidth(120)
            elif combo in (self.remote_output_mode_combo, self.local_output_mode_combo):
                combo.setFixedWidth(92)
            elif combo in (self.remote_lang_combo, self.local_lang_combo):
                combo.setFixedWidth(70)
            else:
                combo.setFixedWidth(90)
            combo.currentIndexChanged.connect(self._notify_settings_changed)
        self.session_mode_combo.currentIndexChanged.connect(self._on_hidden_session_mode_changed)
        self._refresh_asr_backend_hints()
        self.remote_lang_combo.currentIndexChanged.connect(self._on_translation_target_changed)
        self.local_lang_combo.currentIndexChanged.connect(self._on_translation_target_changed)

        # 直接用 grid 布局, 由 resizeEvent 做等分固定尺寸即可
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 1)

        grid.addLayout(
            self._make_panel_header(
                self.remote_original_label,
                self.remote_original_status,
                self.export_remote_original_btn,
                controls=[self.remote_asr_label, self.remote_asr_combo],
            ),
            0,
            0,
        )
        grid.addLayout(
            self._make_panel_header(
                self.remote_translated_label,
                self.remote_translated_status,
                self.export_remote_translated_btn,
                controls=[
                    self.remote_target_label,
                    self.remote_lang_combo,
                    self.remote_output_policy_label,
                    self.remote_output_mode_combo,
                    self.remote_tts_voice_label,
                    self.remote_tts_voice_combo,
                ],
            ),
            0,
            1,
        )
        grid.addWidget(self.remote_original, 1, 0)
        grid.addWidget(self.remote_translated, 1, 1)
        grid.addLayout(
            self._make_panel_header(
                self.local_original_label,
                self.local_original_status,
                self.export_local_original_btn,
                controls=[self.local_asr_label, self.local_asr_combo],
            ),
            2,
            0,
        )
        grid.addLayout(
            self._make_panel_header(
                self.local_translated_label,
                self.local_translated_status,
                self.export_local_translated_btn,
                controls=[
                    self.local_target_label,
                    self.local_lang_combo,
                    self.local_output_policy_label,
                    self.local_output_mode_combo,
                    self.local_tts_voice_label,
                    self.local_tts_voice_combo,
                ],
            ),
            2,
            1,
        )
        grid.addWidget(self.local_original, 3, 0)
        grid.addWidget(self.local_translated, 3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 6)
        mode_row.addWidget(QLabel("音源"))
        mode_row.addWidget(self.meeting_source_combo)
        mode_row.addWidget(self.meeting_device_combo)
        mode_row.addWidget(self.meeting_notice_label)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
        layout.addWidget(self.remote_direct_notice_label)
        layout.addWidget(self.local_direct_notice_label)
        layout.addLayout(grid)
        self._caption_grid = grid

        # 初始正確化 TTS 選項和標籤
        self._on_translation_target_changed()
        self.set_panel_statuses(
            remote_original="idle",
            remote_translated="idle",
            local_original="idle",
            local_translated="idle",
        )
        self._sync_session_mode_buttons()

    def _normalize_asr_language(self, value: str, default: str) -> str:
        normalized = (value or "").strip()
        if normalized == "auto":
            return default
        if normalized in {"none", "auto", "zh-TW", "en", "ja", "ko", "th"}:
            return normalized
        return default

    @staticmethod
    def _config_output_policy(config: AppConfig, channel: str) -> str:
        if channel == "remote":
            value = str(getattr(config.dialogue.remote_to_local, "output_policy", "direct_passthrough") or "direct_passthrough")
            voice = str(getattr(config.runtime, "remote_tts_voice", "") or getattr(config.dialogue, "remote_tts_voice", "") or "")
            tts_enabled = bool(getattr(config.runtime, "remote_tts_enabled", False))
        else:
            value = str(getattr(config.dialogue.local_to_remote, "output_policy", "direct_passthrough") or "direct_passthrough")
            voice = str(getattr(config.runtime, "local_tts_voice", "") or getattr(config.dialogue, "local_tts_voice", "") or "")
            tts_enabled = bool(getattr(config.runtime, "local_tts_enabled", False))
        if value == "direct_passthrough" and (tts_enabled or (voice.strip() and voice.strip().lower() != "none")):
            return "translated_tts"
        return value if value in {"translated_tts", "subtitle_only", "direct_passthrough"} else "subtitle_only"

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._set_target_combo(
                self.session_mode_combo,
                str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
            )
            self._set_target_combo(
                self.meeting_source_combo,
                str(getattr(config.meeting, "audio_source", "system_input") or "system_input"),
            )
            self._set_meeting_device_text(str(getattr(config.meeting, "source_device", "") or ""))
            self._configured_asr_models = {
                "chinese": self._asr_model_label_from_profile(config.asr_channels.local),
                "non_chinese": self._asr_model_label_from_profile(config.asr_channels.remote),
            }
            for channel in _CHANNELS:
                target_value = self._config_translation_target(config, channel)
                voice_value = self._config_tts_voice(config, channel)
                policy_value = self._config_output_policy(config, channel)
                self._set_target_combo(
                    self._channel_combo(channel, "asr"),
                    self._normalize_asr_language(
                        self._config_asr_language(config, channel),
                        self._config_source_language(config, channel),
                    ),
                )
                self._set_target_combo(self._channel_combo(channel, "target"), target_value)
                self._set_target_combo(self._channel_combo(channel, "output"), policy_value)
            self._on_translation_target_changed()
            for channel in _CHANNELS:
                voice_value = self._config_tts_voice(config, channel)
                self._select_combo_data(self._channel_combo(channel, "voice"), voice_value)
                self._ensure_translation_defaults(channel)
        finally:
            self._suspend_notify = False
        self._sync_session_mode_buttons()
        self._update_source_language_controls()
        self._refresh_panel_labels()
        self._refresh_asr_backend_hints()

    def update_config(self, config: AppConfig) -> None:
        config.runtime.session_mode = self.selected_session_mode()
        config.direction.mode = self.selected_mode()
        config.meeting.audio_source = self._get_target_language(self.meeting_source_combo, default="system_input")
        meeting_device = self._get_target_language(self.meeting_device_combo, default="")
        if config.meeting.audio_source == "system_output_loopback":
            config.meeting.output_loopback_device = meeting_device
        else:
            config.meeting.input_device = meeting_device
        enabled_channels: list[str] = []
        for channel in _CHANNELS:
            mode = self.selected_tts_output_mode_for_channel(channel)
            policy = self._selected_output_policy(channel)
            asr_enabled = self._selected_asr_language(channel) != "none"
            raw_target = self._selected_translation_target(channel)
            translation_enabled = asr_enabled and raw_target != "none" and policy != "direct_passthrough"
            target = self._resolved_translation_target(channel)
            self._set_runtime_translation_target(config, channel, raw_target)
            self._set_runtime_translation_enabled(config, channel, translation_enabled)
            if translation_enabled:
                enabled_channels.append(channel)
                self._set_language_target(config, channel, target)
            self._set_runtime_asr_language(
                config,
                channel,
                self._get_target_language(
                    self._channel_combo(channel, "asr"),
                    default=_CHANNEL_DEFAULTS[channel]["asr_default"],
                ),
            )
            self._set_runtime_tts_voice(
                config,
                channel,
                self._selected_tts_voice(channel),
            )
            self._set_runtime_tts_enabled(config, channel, translation_enabled and mode == "tts")

        config.meeting.asr_language = self._selected_asr_language("remote")
        config.meeting.translation_target = self._resolved_translation_target("remote")
        config.meeting.tts_enabled = False
        config.dialogue.remote_asr_language = config.runtime.remote_asr_language
        config.dialogue.local_asr_language = config.runtime.local_asr_language
        config.dialogue.remote_translation_target = config.runtime.remote_translation_target
        config.dialogue.local_translation_target = config.runtime.local_translation_target
        config.dialogue.remote_tts_voice = config.runtime.remote_tts_voice or "none"
        config.dialogue.local_tts_voice = config.runtime.local_tts_voice or "none"
        config.dialogue.remote_to_local.asr_language = config.dialogue.remote_asr_language
        config.dialogue.local_to_remote.asr_language = config.dialogue.local_asr_language
        config.dialogue.remote_to_local.translation_target = config.dialogue.remote_translation_target
        config.dialogue.local_to_remote.translation_target = config.dialogue.local_translation_target
        config.dialogue.remote_to_local.tts_voice = config.dialogue.remote_tts_voice
        config.dialogue.local_to_remote.tts_voice = config.dialogue.local_tts_voice
        config.dialogue.remote_to_local.output_policy = self._selected_output_policy("remote")
        config.dialogue.local_to_remote.output_policy = self._selected_output_policy("local")
        config.runtime.translation_enabled = bool(enabled_channels)
        modes = {channel: self.selected_tts_output_mode_for_channel(channel) for channel in _CHANNELS}
        if any(mode == "tts" for mode in modes.values()):
            config.runtime.tts_output_mode = "tts"
        else:
            config.runtime.tts_output_mode = "passthrough"

    def selected_mode(self) -> str:
        return self.selected_session_mode()

    def selected_session_mode(self) -> str:
        return self._get_target_language(self.session_mode_combo, default="meeting")

    def selected_meeting_source(self) -> str:
        return self._get_target_language(self.meeting_source_combo, default="system_input")

    def selected_meeting_device(self) -> str:
        return self._get_target_language(self.meeting_device_combo, default="")

    def set_session_mode(self, mode: str, *, notify: bool = False) -> None:
        self.session_mode_combo.blockSignals(True)
        try:
            self._set_target_combo(self.session_mode_combo, mode)
        finally:
            self.session_mode_combo.blockSignals(False)
        self._sync_session_mode_buttons()
        self._update_source_language_controls()
        self._refresh_panel_labels()
        if notify:
            self._notify_settings_changed()

    def _on_hidden_session_mode_changed(self, *_args) -> None:
        self._sync_session_mode_buttons()
        self._notify_settings_changed()

    def _sync_session_mode_buttons(self) -> None:
        selected = self.selected_session_mode()
        self.meeting_mode_btn.blockSignals(True)
        self.dialogue_mode_btn.blockSignals(True)
        try:
            self.meeting_mode_btn.setChecked(selected == "meeting")
            self.dialogue_mode_btn.setChecked(selected == "dialogue")
        finally:
            self.meeting_mode_btn.blockSignals(False)
            self.dialogue_mode_btn.blockSignals(False)

    def selected_legacy_direction_mode(self) -> str:
        remote_enabled = self._selected_asr_language("remote") != "none"
        local_enabled = self._selected_asr_language("local") != "none"
        if remote_enabled and local_enabled:
            return "bidirectional"
        if remote_enabled:
            return "meeting_to_local"
        if local_enabled:
            return "local_to_meeting"
        return "bidirectional"

    def selected_asr_language_mode(self) -> str:
        return self._get_target_language(self._channel_combo("remote", "asr"), default="zh-TW")

    def selected_tts_output_mode_for_channel(self, channel: str) -> str:
        if self.selected_session_mode() == "meeting":
            return "subtitle_only"
        policy = self._selected_output_policy(channel)
        if policy == "direct_passthrough":
            return "passthrough"
        if self._selected_asr_language(channel) == "none":
            return "subtitle_only"
        if self._selected_translation_target(channel) == "none":
            return "subtitle_only"
        if policy == "subtitle_only":
            return "subtitle_only"
        selected_voice = self._selected_tts_voice(channel)
        return "tts" if selected_voice != "none" else "subtitle_only"

    def selected_tts_output_mode(self) -> str:
        modes = [self.selected_tts_output_mode_for_channel(channel) for channel in _CHANNELS]
        if any(mode == "tts" for mode in modes):
            return "tts"
        return "subtitle_only" if self.selected_session_mode() == "meeting" else "passthrough"

    def translation_enabled(self, source: str | None = None) -> bool:
        if source in _CHANNELS:
            return self._selected_asr_language(source) != "none" and self._selected_translation_target(source) != "none"
        return any(
            self._selected_asr_language(channel) != "none" and self._selected_translation_target(channel) != "none"
            for channel in _CHANNELS
        )

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def set_direction_controls_enabled(self, enabled: bool) -> None:
        self._direction_controls_enabled = bool(enabled)
        for channel in _CHANNELS:
            self._channel_combo(channel, "asr").setEnabled(self._direction_controls_enabled)
        self._update_source_language_controls()

    def set_panel_statuses(
        self,
        *,
        remote_original: str,
        remote_translated: str,
        local_original: str,
        local_translated: str,
    ) -> None:
        state_map = {
            ("remote", "original"): remote_original,
            ("remote", "translated"): remote_translated,
            ("local", "original"): local_original,
            ("local", "translated"): local_translated,
        }
        for (channel, kind), state in state_map.items():
            self._apply_status(self._channel_status_label(channel, kind), state)

    def set_detected_asr_language(self, source: str, language: str) -> None:
        key = source if source in {"local", "remote"} else "local"
        normalized = (language or "").strip().lower()
        if self._detected_asr_language.get(key, "") == normalized:
            return
        self._detected_asr_language[key] = normalized
        self._refresh_panel_labels()

    def _handle_start_clicked(self) -> None:
        if self._on_start_clicked:
            self._on_start_clicked()

    def _notify_settings_changed(self, *_args) -> None:
        if self._suspend_notify:
            return
        self._refresh_meeting_device_combo()
        for channel in _CHANNELS:
            self._ensure_translation_defaults(channel)
        self._update_source_language_controls()
        self._refresh_panel_labels()
        self._refresh_asr_backend_hints()
        if self._on_output_mode_changed:
            mode = self.selected_tts_output_mode()
            if mode != self._last_emitted_output_mode:
                self._last_emitted_output_mode = mode
                self._on_output_mode_changed(mode)
        if self._on_settings_changed:
            self._on_settings_changed()

    def _refresh_asr_backend_hints(self) -> None:
        for channel in _CHANNELS:
            combo = self._channel_combo(channel, "asr")
            hint = self._backend_hint_for_asr_language(self._selected_asr_language(channel))
            combo.setToolTip(hint)
            combo.setStatusTip(hint)

    @staticmethod
    def _backend_hint_for_asr_language(language: str) -> str:
        normalized = (language or "").strip().lower().replace("_", "-")
        if normalized == "none":
            return "此通道已停用 ASR。"
        if normalized == "auto":
            return "auto 目前固定走 faster-whisper。"
        return "此通道會使用 faster-whisper。"

    def _handle_clear_clicked(self) -> None:
        self.clear()
        if self._on_clear_clicked:
            self._on_clear_clicked()

    def _on_translation_target_changed(self, *_args) -> None:
        for channel in _CHANNELS:
            target = self._get_target_language(self._channel_combo(channel, "target"), default="none")
            self._populate_tts_voice_combo(self._channel_combo(channel, "voice"), target)
            self._ensure_translation_defaults(channel)
        self._notify_settings_changed()

    def _populate_tts_voice_combo(self, combo: QComboBox, target: str) -> None:
        current_value = self._get_target_language(combo, default="none")
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("無", "none")
            choices: list[tuple[str, str]] = []
            normalized_target = (target or "").strip().lower().replace("_", "-")

            if normalized_target == "none":
                combo.setEnabled(False)
                combo.setFixedWidth(80)
            else:
                combo.setEnabled(True)
                combo.setFixedWidth(130)
                effective_key = ""
                if normalized_target in {"zh-tw", "zh"}:
                    effective_key = "zh-TW"
                elif normalized_target in {"en", "eng"}:
                    effective_key = "en"
                elif normalized_target in {"ja", "jp"}:
                    effective_key = "ja"
                elif normalized_target in {"ko", "kr"}:
                    effective_key = "ko"
                elif normalized_target in {"th", "thai"}:
                    effective_key = "th"

                if effective_key:
                    choices = _TTS_VOICE_OPTIONS.get(effective_key, [])
                if not choices:
                    for group in _TTS_VOICE_OPTIONS.values():
                        choices.extend(group)

            for voice_value, voice_label in choices:
                combo.addItem(voice_label, voice_value)

            # Ensure popup width can容納長字串，不會被截斷
            view = combo.view()
            if view is not None:
                font_metrics = combo.fontMetrics()
                width = max((font_metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=combo.width()) + 40
                view.setMinimumWidth(width)
                combo.setMinimumWidth(min(max(width, combo.minimumWidth()), 500))

            chosen = current_value if current_value != "none" else "none"
            idx = combo.findData(chosen)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def clear(self) -> None:
        for channel in _CHANNELS:
            for kind in ("original", "translated"):
                editor = self._channel_editor(channel, kind)
                editor.clear()
                self._editor_committed[editor] = []
                self._editor_has_partial[editor] = False

    def set_remote_original_lines(self, lines: list[str]) -> None:
        self._set_channel_lines("remote", "original", lines)

    def set_remote_translated_lines(self, lines: list[str]) -> None:
        self._set_channel_lines("remote", "translated", lines)

    def set_local_original_lines(self, lines: list[str]) -> None:
        self._set_channel_lines("local", "original", lines)

    def set_local_translated_lines(self, lines: list[str]) -> None:
        self._set_channel_lines("local", "translated", lines)

    def _set_channel_lines(self, channel: str, kind: str, lines: list[str]) -> None:
        self._set_editor_lines(self._channel_editor(channel, kind), lines)

    def _set_editor_lines(self, editor: QTextEdit, lines: list[str]) -> None:
        # 新訊息先顯示，舊訊息保留不刪除
        final_lines = [line for line in lines if not line.startswith("[partial]")]
        partial_line: str | None = next((line for line in lines if line.startswith("[partial]")), None)

        if partial_line is not None:
            # partial 應該展示在最上方，便於使用者看到正在輸入中的句子
            full_lines = [partial_line] + final_lines
        else:
            full_lines = final_lines

        editor.setPlainText("\n".join(full_lines))

        self._editor_committed[editor] = final_lines
        self._editor_has_partial[editor] = partial_line is not None

        scrollbar = editor.verticalScrollBar()
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.minimum()))

    @staticmethod
    def _append_editor_line(editor: QTextEdit, line: str) -> None:
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if editor.document().isEmpty():
            cursor.insertText(line)
        else:
            cursor.insertText("\n" + line)
        editor.setTextCursor(cursor)

    @staticmethod
    def _remove_editor_last_line(editor: QTextEdit) -> None:
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        editor.setTextCursor(cursor)

    def _export_editor(self, editor: QTextEdit, panel_name: str) -> None:
        default_name = f"caption_{panel_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "匯出字幕", default_name, "文字檔 (*.txt)")
        if not path:
            return
        text = self._export_text_for_editor(editor)
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(text)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "匯出失敗", str(exc))

    @staticmethod
    def _export_text_for_editor(editor: QTextEdit) -> str:
        lines = editor.toPlainText().splitlines()
        exported: list[str] = []
        for line in reversed(lines):
            if line.startswith("[final]"):
                exported.append(line[len("[final]"):].lstrip())
            else:
                exported.append(line)
        return "\n".join(exported)

    @staticmethod
    def _get_target_language(combo: QComboBox, default: str) -> str:
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        return default

    @staticmethod
    def _set_target_combo(combo: QComboBox, target: str) -> None:
        idx = combo.findData(target)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _resolved_translation_target(self, channel: str) -> str:
        selected = self._selected_translation_target(channel)
        if selected != "none":
            return selected
        return _CHANNEL_DEFAULTS[channel]["target_default"]

    def _resolved_tts_voice(self, channel: str) -> str:
        selected = self._selected_tts_voice(channel)
        if selected:
            return selected
        return default_voice_for_language(self._resolved_translation_target(channel)) or "none"

    def _selected_translation_target(self, channel: str) -> str:
        return self._get_target_language(self._channel_combo(channel, "target"), default="none")

    def _selected_asr_language(self, channel: str) -> str:
        return self._get_target_language(self._channel_combo(channel, "asr"), default="none")

    def _selected_output_policy(self, channel: str) -> str:
        value = self._get_target_language(self._channel_combo(channel, "output"), default="subtitle_only")
        return value if value in {"translated_tts", "subtitle_only", "direct_passthrough"} else "subtitle_only"

    def _selected_tts_voice(self, channel: str) -> str:
        return self._get_target_language(self._channel_combo(channel, "voice"), default="none")

    def _ensure_translation_defaults(self, channel: str) -> None:
        if (
            self._selected_output_policy(channel) != "translated_tts"
            or self._selected_asr_language(channel) == "none"
            or self._selected_translation_target(channel) == "none"
        ):
            return
        target_combo = self._channel_combo(channel, "target")
        voice_combo = self._channel_combo(channel, "voice")
        target = self._get_target_language(target_combo, default="none")
        if target == "none":
            self._populate_tts_voice_combo(voice_combo, target)
            return
        if self._selected_tts_voice(channel) == "none":
            self._select_combo_data(voice_combo, self._resolved_tts_voice(channel))

    def _update_source_language_controls(self) -> None:
        enabled = self._direction_controls_enabled
        meeting_mode = self.selected_session_mode() == "meeting"
        self.meeting_source_combo.setVisible(meeting_mode)
        self.meeting_device_combo.setVisible(meeting_mode)
        self.meeting_notice_label.setVisible(meeting_mode)
        self.remote_tts_voice_label.setVisible(not meeting_mode)
        self.remote_tts_voice_combo.setVisible(not meeting_mode)
        self.remote_output_policy_label.setVisible(not meeting_mode)
        self.remote_output_mode_combo.setVisible(not meeting_mode)
        for widget in (
            self.local_original,
            self.local_translated,
            self.local_original_label,
            self.local_translated_label,
            self.local_original_status,
            self.local_translated_status,
            self.export_local_original_btn,
            self.export_local_translated_btn,
            self.local_asr_label,
            self.local_asr_combo,
            self.local_target_label,
            self.local_lang_combo,
            self.local_output_policy_label,
            self.local_output_mode_combo,
            self.local_tts_voice_label,
            self.local_tts_voice_combo,
        ):
            widget.setVisible(not meeting_mode)
        if hasattr(self, "_caption_grid"):
            self._caption_grid.setRowStretch(1, 1)
            self._caption_grid.setRowStretch(3, 0 if meeting_mode else 1)
        for channel in _CHANNELS:
            if meeting_mode and channel == "local":
                self._channel_combo(channel, "asr").setEnabled(False)
                self._channel_combo(channel, "target").setEnabled(False)
                self._channel_combo(channel, "output").setEnabled(False)
                self._channel_combo(channel, "voice").setEnabled(False)
                continue
            policy = self._selected_output_policy(channel)
            direct = (not meeting_mode) and policy == "direct_passthrough"
            asr_enabled = enabled and not direct and self._selected_asr_language(channel) != "none"
            translation_enabled = asr_enabled and self._selected_translation_target(channel) != "none"
            self._channel_combo(channel, "asr").setEnabled(enabled and not direct)
            self._channel_combo(channel, "target").setEnabled(asr_enabled)
            self._channel_combo(channel, "output").setEnabled(enabled and not meeting_mode)
            self._channel_combo(channel, "voice").setEnabled(translation_enabled and policy == "translated_tts")
            self._channel_combo(channel, "asr").setToolTip("ASR language: zh-TW / en / ja / ko / th")
            self._channel_combo(channel, "target").setToolTip("Translation target language")
            self._channel_combo(channel, "output").setToolTip("翻譯語音會輸出 TTS；字幕會辨識翻譯但不輸出聲音；直通不辨識、不翻譯、不記錄。")
            self._channel_combo(channel, "voice").setToolTip("TTS voice for translated speech")
        self.remote_direct_notice_label.setVisible((not meeting_mode) and self._selected_output_policy("remote") == "direct_passthrough")
        self.local_direct_notice_label.setVisible((not meeting_mode) and self._selected_output_policy("local") == "direct_passthrough")
        self._apply_editor_layout_constraints()

    def set_meeting_devices(self, input_devices: list[str], output_devices: list[str]) -> None:
        self._meeting_input_devices = list(input_devices)
        self._meeting_output_devices = list(output_devices)
        self._refresh_meeting_device_combo()

    def _refresh_meeting_device_combo(self) -> None:
        current = self._get_target_language(self.meeting_device_combo, default="")
        source_kind = self._get_target_language(self.meeting_source_combo, default="system_input")
        devices = getattr(self, "_meeting_output_devices", []) if source_kind == "system_output_loopback" else getattr(self, "_meeting_input_devices", [])
        self.meeting_device_combo.blockSignals(True)
        try:
            self.meeting_device_combo.clear()
            self.meeting_device_combo.addItem("", "")
            for name in devices:
                self.meeting_device_combo.addItem(name, name)
            self._set_target_combo(self.meeting_device_combo, current)
        finally:
            self.meeting_device_combo.blockSignals(False)

    def _set_meeting_device_text(self, value: str) -> None:
        if not value:
            return
        if self.meeting_device_combo.findData(value) < 0:
            self.meeting_device_combo.addItem(value, value)
        self._set_target_combo(self.meeting_device_combo, value)

    def _refresh_panel_labels(self) -> None:
        meeting_mode = self.selected_session_mode() == "meeting"
        for channel in _CHANNELS:
            original = _CHANNEL_DEFAULTS[channel]["original_label"]
            translated_text = (
                _CHANNEL_DEFAULTS[channel]["translated_label"]
                if self.translation_enabled(channel)
                else _CHANNEL_DEFAULTS[channel]["output_label"]
            )
            if meeting_mode and channel == "remote":
                original = "會議原文"
                translated_text = "會議翻譯"
            self._channel_text_label(channel, "original").setText(
                f"{original}（{self._channel_asr_model_label(channel)}）"
            )
            self._channel_text_label(channel, "translated").setText(translated_text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_editor_layout_constraints()
        self._refresh_panel_labels()

    def _apply_editor_layout_constraints(self) -> None:
        meeting_mode = self.selected_session_mode() == "meeting"
        if meeting_mode:
            for editor in (self.remote_original, self.remote_translated):
                editor.setMinimumSize(0, 240)
                editor.setMaximumSize(16777215, 16777215)
            for editor in (self.local_original, self.local_translated):
                editor.setMinimumSize(0, 0)
                editor.setMaximumSize(16777215, 0)
        else:
            for editor in (self.remote_original, self.remote_translated, self.local_original, self.local_translated):
                editor.setMinimumSize(0, 160)
                editor.setMaximumSize(16777215, 16777215)

    def update_translation_voice_labels(self, config: AppConfig) -> None:
        _ = config
        self._refresh_panel_labels()

    def _channel_combo(self, channel: str, kind: str) -> QComboBox:
        if channel == "remote":
            mapping = {
                "asr": self.remote_asr_combo,
                "target": self.remote_lang_combo,
                "output": self.remote_output_mode_combo,
                "voice": self.remote_tts_voice_combo,
            }
        else:
            mapping = {
                "asr": self.local_asr_combo,
                "target": self.local_lang_combo,
                "output": self.local_output_mode_combo,
                "voice": self.local_tts_voice_combo,
            }
        return mapping[kind]

    def _channel_editor(self, channel: str, kind: str) -> QTextEdit:
        if channel == "remote":
            return self.remote_original if kind == "original" else self.remote_translated
        return self.local_original if kind == "original" else self.local_translated

    def _channel_text_label(self, channel: str, kind: str) -> QLabel:
        if channel == "remote":
            return self.remote_original_label if kind == "original" else self.remote_translated_label
        return self.local_original_label if kind == "original" else self.local_translated_label

    def _channel_status_label(self, channel: str, kind: str) -> QLabel:
        if channel == "remote":
            return self.remote_original_status if kind == "original" else self.remote_translated_status
        return self.local_original_status if kind == "original" else self.local_translated_status

    def _channel_asr_model_label(self, channel: str) -> str:
        selected_language = self._selected_asr_language(channel)
        family = asr_profile_family_for_language(selected_language)
        if family == "disabled":
            return "停用"
        return self._configured_asr_models.get(family, "large-v3-turbo")

    def _make_panel_header(
        self,
        title_label: QLabel,
        status_label: QLabel,
        export_btn: QPushButton | None = None,
        controls: list[QWidget] | None = None,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(title_label)
        if controls:
            for control in controls:
                row.addWidget(control)
        row.addStretch(1)
        row.addWidget(status_label)
        if export_btn is not None:
            row.addWidget(export_btn)
        return row

    @staticmethod
    def _apply_status(label: QLabel, state: str) -> None:
        normalized = state if state in _STATUS_TEXT else "idle"
        color = _STATUS_DOT[normalized]
        label.setText(f"<span style='color:{color}'>●</span> {_STATUS_TEXT[normalized]}")

    @staticmethod
    def _language_label(language: str) -> str:
        normalized = (language or "").strip().lower()
        if not normalized:
            return "未設定"
        if normalized in _LANG_LABELS:
            return _LANG_LABELS[normalized]
        if "-" in normalized:
            short = normalized.split("-", 1)[0]
            if short in _LANG_LABELS:
                return _LANG_LABELS[short]
        return normalized

    @staticmethod
    def _configure_combo_popup(combo: QComboBox) -> None:
        # 顯示多行選項，並在需要時啟用捲軸，避免只顯示1行導致無法選擇其他條目。
        combo.setMaxVisibleItems(8)
        view = combo.view()
        if view is None:
            return
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    @staticmethod
    def _display_voice_label(voice_name: str) -> str:
        value = (voice_name or "").strip()
        return value or "未設定"
