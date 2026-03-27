from __future__ import annotations

from datetime import datetime
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
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
    QSplitter,
)

from app.infra.config.schema import AppConfig
from app.infra.tts.voice_policy import resolve_edge_voice_for_target

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

_SUPPORTED_LANGUAGES: list[tuple[str, str]] = [
    ("zh-TW", "中文"),
    ("en", "英文"),
    ("ja", "日文"),
    ("ko", "韓文"),
    ("th", "泰文"),
]

_LANG_LABELS: dict[str, str] = {
    "en": "英文",
    "zh": "中文",
    "zh-tw": "中文",
    "ja": "日文",
    "ko": "韓文",
    "th": "泰文",
}

_ASR_CHOICES: list[tuple[str, str]] = [
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


def _build_target_choices() -> list[tuple[str, str]]:
    return [(code, label) for code, label in _SUPPORTED_LANGUAGES]


class LiveCaptionPage(QWidget):
    def __init__(
        self,
        on_clear_clicked: Callable[[], None] | None = None,
        on_start_clicked: Callable[[], None] | None = None,
        on_test_local_tts_clicked: Callable[[], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
        on_output_mode_changed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_clear_clicked = on_clear_clicked
        self._on_start_clicked = on_start_clicked
        self._on_test_local_tts_clicked = on_test_local_tts_clicked
        self._on_settings_changed = on_settings_changed
        self._on_output_mode_changed = on_output_mode_changed
        self._suspend_notify = False
        self._direction_controls_enabled = True
        self._detected_asr_language: dict[str, str] = {"local": "", "remote": ""}

        self.remote_asr_combo = QComboBox()
        self.local_asr_combo = QComboBox()
        for code, label in _ASR_CHOICES:
            self.remote_asr_combo.addItem(label, code)
            self.local_asr_combo.addItem(label, code)
        self._configure_combo_popup(self.remote_asr_combo)
        self._configure_combo_popup(self.local_asr_combo)

        self.remote_lang_combo = QComboBox()
        self.local_lang_combo = QComboBox()
        for code, label in _TRANSLATION_TARGET_CHOICES:
            self.remote_lang_combo.addItem(label, code)
            self.local_lang_combo.addItem(label, code)
        self._configure_combo_popup(self.remote_lang_combo)
        self._configure_combo_popup(self.local_lang_combo)

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
        self.test_local_tts_btn = QPushButton("測試TTS")
        self.test_local_tts_btn.setToolTip("會用本地輸出語言產生測試語音，播放到本地輸出裝置。")
        self.test_local_tts_btn.clicked.connect(self._handle_test_local_tts_clicked)

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
            self.remote_asr_combo,
            self.local_asr_combo,
            self.remote_lang_combo,
            self.local_lang_combo,
            self.remote_tts_voice_combo,
            self.local_tts_voice_combo,
        ):
            combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            if combo in (self.remote_tts_voice_combo, self.local_tts_voice_combo):
                combo.setFixedWidth(120)
            elif combo in (self.remote_lang_combo, self.local_lang_combo):
                combo.setFixedWidth(70)
            else:
                combo.setFixedWidth(90)
            combo.currentIndexChanged.connect(self._notify_settings_changed)
        self.remote_lang_combo.currentIndexChanged.connect(self._on_translation_target_changed)
        self.local_lang_combo.currentIndexChanged.connect(self._on_translation_target_changed)

        # 用 Splitter 保證四個區域等分 (2x2) ，避免右邊被額外控件寬度擠大
        top_left_widget = QWidget()
        tl_layout = QVBoxLayout(top_left_widget)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(2)
        tl_layout.addLayout(self._make_panel_header(self.remote_original_label, self.remote_original_status, self.export_remote_original_btn, controls=[QLabel("ASR語言"), self.remote_asr_combo]))
        tl_layout.addWidget(self.remote_original)

        top_right_widget = QWidget()
        tr_layout = QVBoxLayout(top_right_widget)
        tr_layout.setContentsMargins(0, 0, 0, 0)
        tr_layout.setSpacing(2)
        tr_layout.addLayout(self._make_panel_header(self.remote_translated_label, self.remote_translated_status, self.export_remote_translated_btn, controls=[QLabel("目標"), self.remote_lang_combo, QLabel("TTS語音"), self.remote_tts_voice_combo]))
        tr_layout.addWidget(self.remote_translated)

        bottom_left_widget = QWidget()
        bl_layout = QVBoxLayout(bottom_left_widget)
        bl_layout.setContentsMargins(0, 0, 0, 0)
        bl_layout.setSpacing(2)
        bl_layout.addLayout(self._make_panel_header(self.local_original_label, self.local_original_status, self.export_local_original_btn, controls=[QLabel("ASR語言"), self.local_asr_combo]))
        bl_layout.addWidget(self.local_original)

        bottom_right_widget = QWidget()
        br_layout = QVBoxLayout(bottom_right_widget)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.setSpacing(2)
        br_layout.addLayout(self._make_panel_header(self.local_translated_label, self.local_translated_status, self.export_local_translated_btn, controls=[QLabel("目標"), self.local_lang_combo, QLabel("TTS語音"), self.local_tts_voice_combo, self.test_local_tts_btn]))
        br_layout.addWidget(self.local_translated)

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
                controls=[QLabel("ASR語言"), self.remote_asr_combo],
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
                    QLabel("翻譯目標"),
                    self.remote_lang_combo,
                    QLabel("TTS語音"),
                    self.remote_tts_voice_combo,
                    self.test_local_tts_btn,
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
                controls=[QLabel("ASR語言"), self.local_asr_combo],
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
                    QLabel("翻譯目標"),
                    self.local_lang_combo,
                    QLabel("TTS語音"),
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
        layout.addLayout(grid)

        # 初始正確化 TTS 選項和標籤
        self._on_translation_target_changed()
        self.set_panel_statuses(
            remote_original="idle",
            remote_translated="idle",
            local_original="idle",
            local_translated="idle",
        )

    def _normalize_asr_language(self, value: str, default: str) -> str:
        normalized = (value or "").strip()
        if normalized in {"zh-TW", "en", "ja", "ko", "th"}:
            return normalized
        if normalized.lower() == "auto":
            return default
        return default

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._set_target_combo(
                self.remote_asr_combo,
                self._normalize_asr_language(
                    str(getattr(config.runtime, "remote_asr_language", "")),
                    str(getattr(config.language, "meeting_source", "zh-TW") or "zh-TW"),
                ),
            )
            self._set_target_combo(
                self.local_asr_combo,
                self._normalize_asr_language(
                    str(getattr(config.runtime, "local_asr_language", "")),
                    str(getattr(config.language, "local_source", "en") or "en"),
                ),
            )
            self._set_target_combo(
                self.remote_lang_combo,
                str(getattr(config.runtime, "remote_translation_target", config.language.meeting_target or "zh-TW") or "none"),
            )
            self._set_target_combo(
                self.local_lang_combo,
                str(getattr(config.runtime, "local_translation_target", config.language.local_target or "en") or "none"),
            )
            self._on_translation_target_changed()
            self._select_combo_data(
                self.remote_tts_voice_combo,
                str(getattr(config.runtime, "remote_tts_voice", "none") or "none"),
            )
            self._select_combo_data(
                self.local_tts_voice_combo,
                str(getattr(config.runtime, "local_tts_voice", "none") or "none"),
            )
        finally:
            self._suspend_notify = False
        self._update_source_language_controls()
        self._refresh_panel_labels()

    def update_config(self, config: AppConfig) -> None:
        config.direction.mode = "bidirectional"

        remote_translation_target = self._get_target_language(self.remote_lang_combo, default="none")
        local_translation_target = self._get_target_language(self.local_lang_combo, default="none")
        config.runtime.remote_translation_target = remote_translation_target
        config.runtime.local_translation_target = local_translation_target
        config.runtime.remote_translation_enabled = remote_translation_target != "none"
        config.runtime.local_translation_enabled = local_translation_target != "none"
        config.runtime.translation_enabled = config.runtime.remote_translation_enabled or config.runtime.local_translation_enabled

        if config.runtime.remote_translation_enabled:
            config.language.meeting_target = remote_translation_target
        if config.runtime.local_translation_enabled:
            config.language.local_target = local_translation_target

        config.runtime.remote_asr_language = self._get_target_language(self.remote_asr_combo, default="zh-TW")
        config.runtime.local_asr_language = self._get_target_language(self.local_asr_combo, default="en")

        config.runtime.remote_tts_voice = self._get_target_language(self.remote_tts_voice_combo, default="none")
        config.runtime.local_tts_voice = self._get_target_language(self.local_tts_voice_combo, default="none")

        config.runtime.remote_tts_enabled = config.runtime.remote_tts_voice != "none" and config.runtime.remote_translation_enabled
        config.runtime.local_tts_enabled = config.runtime.local_tts_voice != "none" and config.runtime.local_translation_enabled

        if not config.runtime.remote_translation_enabled and not config.runtime.local_translation_enabled:
            config.runtime.tts_output_mode = "passthrough"
        elif config.runtime.remote_tts_enabled or config.runtime.local_tts_enabled:
            config.runtime.tts_output_mode = "tts"
        else:
            config.runtime.tts_output_mode = "subtitle_only"

    def selected_mode(self) -> str:
        return "bidirectional"

    def selected_asr_language_mode(self) -> str:
        return self._get_target_language(self.remote_asr_combo, default="auto")

    def selected_tts_output_mode_for_channel(self, channel: str) -> str:
        if channel == "remote":
            target = self._get_target_language(self.remote_lang_combo, default="none")
            voice = self._get_target_language(self.remote_tts_voice_combo, default="none")
        else:
            target = self._get_target_language(self.local_lang_combo, default="none")
            voice = self._get_target_language(self.local_tts_voice_combo, default="none")

        if target == "none":
            return "passthrough"
        if voice == "none":
            return "subtitle_only"
        return "tts"

    def selected_tts_output_mode(self) -> str:
        remote_mode = self.selected_tts_output_mode_for_channel("remote")
        local_mode = self.selected_tts_output_mode_for_channel("local")
        if remote_mode == "tts" or local_mode == "tts":
            return "tts"
        if remote_mode == "passthrough" and local_mode == "passthrough":
            return "passthrough"
        return "subtitle_only"

    def translation_enabled(self, source: str | None = None) -> bool:
        if source == "remote":
            return self._get_target_language(self.remote_lang_combo, default="none") != "none"
        if source == "local":
            return self._get_target_language(self.local_lang_combo, default="none") != "none"
        return (
            self._get_target_language(self.remote_lang_combo, default="none") != "none"
            or self._get_target_language(self.local_lang_combo, default="none") != "none"
        )

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def set_direction_controls_enabled(self, enabled: bool) -> None:
        self._direction_controls_enabled = bool(enabled)
        for widget in (
            self.remote_asr_combo,
            self.local_asr_combo,
            self.remote_lang_combo,
            self.local_lang_combo,
            self.remote_tts_voice_combo,
            self.local_tts_voice_combo,
        ):
            widget.setEnabled(self._direction_controls_enabled)
        self._update_source_language_controls()

    def set_panel_statuses(
        self,
        *,
        remote_original: str,
        remote_translated: str,
        local_original: str,
        local_translated: str,
    ) -> None:
        self._apply_status(self.remote_original_status, remote_original)
        self._apply_status(self.remote_translated_status, remote_translated)
        self._apply_status(self.local_original_status, local_original)
        self._apply_status(self.local_translated_status, local_translated)

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
        self._update_source_language_controls()
        self._refresh_panel_labels()
        if self._on_settings_changed:
            self._on_settings_changed()

    def _handle_clear_clicked(self) -> None:
        self.clear()
        if self._on_clear_clicked:
            self._on_clear_clicked()

    def _handle_test_local_tts_clicked(self) -> None:
        if self._on_test_local_tts_clicked:
            self._on_test_local_tts_clicked()

    def _on_translation_target_changed(self, *_args) -> None:
        remote_target = self._get_target_language(self.remote_lang_combo, default="none")
        local_target = self._get_target_language(self.local_lang_combo, default="none")
        self._populate_tts_voice_combo(self.remote_tts_voice_combo, remote_target)
        self._populate_tts_voice_combo(self.local_tts_voice_combo, local_target)
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

    def _handle_output_mode_changed(self, *_args) -> None:
        if self._on_output_mode_changed:
            self._on_output_mode_changed(self.selected_tts_output_mode())

    def clear(self) -> None:
        for editor in (self.remote_original, self.remote_translated, self.local_original, self.local_translated):
            editor.clear()
            self._editor_committed[editor] = []
            self._editor_has_partial[editor] = False

    def set_remote_original_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.remote_original, lines)

    def set_remote_translated_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.remote_translated, lines)

    def set_local_original_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.local_original, lines)

    def set_local_translated_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.local_translated, lines)

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
        text = editor.toPlainText()
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(text)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "匯出失敗", str(exc))

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

    def _update_source_language_controls(self) -> None:
        enabled = self._direction_controls_enabled
        self.remote_asr_combo.setEnabled(enabled)
        self.local_asr_combo.setEnabled(enabled)
        self.remote_lang_combo.setEnabled(enabled)
        self.local_lang_combo.setEnabled(enabled)
        self.remote_tts_voice_combo.setEnabled(enabled)
        self.local_tts_voice_combo.setEnabled(enabled)

        self.remote_asr_combo.setToolTip("ASR辨識語言，可選 auto 或指定語言")
        self.local_asr_combo.setToolTip("ASR辨識語言，可選 auto 或指定語言")
        self.remote_lang_combo.setToolTip("LLM翻譯目標：無 = 原音直通")
        self.local_lang_combo.setToolTip("LLM翻譯目標：無 = 原音直通")
        self.remote_tts_voice_combo.setToolTip("TTS輸出聲線：無 = 失聲")
        self.local_tts_voice_combo.setToolTip("TTS輸出聲線：無 = 失聲")

    def _refresh_panel_labels(self) -> None:
        remote_target = self._get_target_language(self.remote_lang_combo, default="zh-TW")
        local_target = self._get_target_language(self.local_lang_combo, default="en")
        remote_detected = self._detected_asr_language.get("remote", "")
        local_detected = self._detected_asr_language.get("local", "")

        remote_asr = self._get_target_language(self.remote_asr_combo, default="auto")
        local_asr = self._get_target_language(self.local_asr_combo, default="auto")

        if remote_asr != "auto":
            remote_source_text = f"手動：{self._language_label(remote_asr)}"
        elif remote_detected:
            remote_source_text = f"自動偵測：{self._language_label(remote_detected)}"
        else:
            remote_source_text = "自動偵測"

        if local_asr != "auto":
            local_source_text = f"手動：{self._language_label(local_asr)}"
        elif local_detected:
            local_source_text = f"自動偵測：{self._language_label(local_detected)}"
        else:
            local_source_text = "自動偵測"

        self.remote_original_label.setText(f"遠端原文（{remote_source_text} ASR）")
        self.local_original_label.setText(f"本地原文（{local_source_text} ASR）")

        if self.translation_enabled("remote"):
            self.remote_translated_label.setText("遠端翻譯")
        else:
            self.remote_translated_label.setText("遠端輸出")

        if self.translation_enabled("local"):
            self.local_translated_label.setText("本地翻譯")
        else:
            self.local_translated_label.setText("本地輸出")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        totalw = max(1, self.width())
        totalh = max(1, self.height())

        # 上方控件高度大約 55（可視），留空再分配給 4 個區域
        header_height = 80
        content_w = totalw - 12
        content_h = max(1, totalh - header_height)

        cell_w = content_w // 2
        cell_h = content_h // 2

        for editor in (self.remote_original, self.remote_translated, self.local_original, self.local_translated):
            editor.setMinimumSize(cell_w, cell_h)
            editor.setMaximumSize(cell_w, cell_h)

        if self.translation_enabled("local"):
            self.local_translated_label.setText("本地翻譯")
        else:
            self.local_translated_label.setText("本地輸出")

    def update_translation_voice_labels(self, config: AppConfig) -> None:
        self._refresh_panel_labels()

        remote_voice = str(getattr(config.runtime, "remote_tts_voice", "none") or "none")
        local_voice = str(getattr(config.runtime, "local_tts_voice", "none") or "none")

        if self.translation_enabled("remote"):
            self.remote_translated_label.setText("遠端翻譯")
        elif remote_voice != "none":
            self.remote_translated_label.setText("原音直通")
        else:
            self.remote_translated_label.setText("遠端輸出")

        if self.translation_enabled("local"):
            self.local_translated_label.setText("本地翻譯")
        elif local_voice != "none":
            self.local_translated_label.setText("本地輸出")
        else:
            self.local_translated_label.setText("本地輸出")

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
