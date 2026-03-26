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

        self.remote_lang_combo = QComboBox()
        self.local_lang_combo = QComboBox()
        for code, label in _build_target_choices():
            self.remote_lang_combo.addItem(label, code)
            self.local_lang_combo.addItem(label, code)
        self._configure_combo_popup(self.remote_lang_combo)
        self._configure_combo_popup(self.local_lang_combo)

        self.remote_translation_enabled_check = QCheckBox("啟用遠端翻譯")
        self.remote_translation_enabled_check.setChecked(True)
        self.remote_translation_enabled_check.setToolTip("關閉時略過遠端 LLM，直接顯示遠端 ASR 結果")
        self.local_translation_enabled_check = QCheckBox("啟用本地翻譯")
        self.local_translation_enabled_check.setChecked(True)
        self.local_translation_enabled_check.setToolTip("關閉時略過本地 LLM，直接顯示本地 ASR 結果")

        self.asr_language_label = QLabel("ASR 語言：自動偵測")
        self.asr_language_label.setToolTip("ASR 只使用自動語言偵測，不再提供手動語言固定。")

        self.tts_output_mode_combo = QComboBox()
        self.tts_output_mode_combo.addItem("翻譯 + TTS", "tts")
        self.tts_output_mode_combo.addItem("翻譯 + 字幕", "subtitle_only")
        self.tts_output_mode_combo.addItem("原音直通", "passthrough")
        self._configure_combo_popup(self.tts_output_mode_combo)

        self.start_btn = QPushButton("開始")
        self.start_btn.clicked.connect(self._handle_start_clicked)
        self.clear_btn = QPushButton("清空字幕")
        self.clear_btn.clicked.connect(self._handle_clear_clicked)
        self.test_local_tts_btn = QPushButton("測試本地輸出TTS")
        self.test_local_tts_btn.setToolTip("會用遠端翻譯語言產生測試語音，播放到本地輸出裝置。")
        self.test_local_tts_btn.clicked.connect(self._handle_test_local_tts_clicked)

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
            editor.setMinimumHeight(220)
            editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for combo in (self.remote_lang_combo, self.local_lang_combo, self.tts_output_mode_combo):
            combo.currentIndexChanged.connect(self._notify_settings_changed)
        self.remote_translation_enabled_check.toggled.connect(self._notify_settings_changed)
        self.local_translation_enabled_check.toggled.connect(self._notify_settings_changed)
        self.tts_output_mode_combo.currentIndexChanged.connect(self._handle_output_mode_changed)

        top_row_primary = QHBoxLayout()
        top_row_primary.setContentsMargins(0, 0, 0, 0)
        top_row_primary.setSpacing(8)
        top_row_primary.addWidget(QLabel("遠端翻譯語言"))
        top_row_primary.addWidget(self.remote_lang_combo, 0)
        top_row_primary.addSpacing(8)
        top_row_primary.addWidget(QLabel("本地翻譯語言"))
        top_row_primary.addWidget(self.local_lang_combo, 0)
        top_row_primary.addStretch(1)

        top_row_secondary = QHBoxLayout()
        top_row_secondary.setContentsMargins(0, 0, 0, 0)
        top_row_secondary.setSpacing(8)
        top_row_secondary.addWidget(self.remote_translation_enabled_check, 0)
        top_row_secondary.addWidget(self.local_translation_enabled_check, 0)
        top_row_secondary.addWidget(self.asr_language_label, 0)
        top_row_secondary.addWidget(QLabel("輸出模式"))
        top_row_secondary.addWidget(self.tts_output_mode_combo, 0)
        top_row_secondary.addWidget(self.test_local_tts_btn, 0)
        top_row_secondary.addStretch(1)
        top_row_secondary.addWidget(self.clear_btn, 0)
        top_row_secondary.addWidget(self.start_btn, 0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.addLayout(self._make_panel_header(self.remote_original_label, self.remote_original_status, self.export_remote_original_btn), 0, 0)
        grid.addLayout(self._make_panel_header(self.remote_translated_label, self.remote_translated_status, self.export_remote_translated_btn), 0, 1)
        grid.addWidget(self.remote_original, 1, 0)
        grid.addWidget(self.remote_translated, 1, 1)
        grid.addLayout(self._make_panel_header(self.local_original_label, self.local_original_status, self.export_local_original_btn), 2, 0)
        grid.addLayout(self._make_panel_header(self.local_translated_label, self.local_translated_status, self.export_local_translated_btn), 2, 1)
        grid.addWidget(self.local_original, 3, 0)
        grid.addWidget(self.local_translated, 3, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(top_row_primary)
        layout.addLayout(top_row_secondary)
        layout.addLayout(grid)
        self.set_panel_statuses(
            remote_original="idle",
            remote_translated="idle",
            local_original="idle",
            local_translated="idle",
        )

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._set_target_combo(self.remote_lang_combo, config.language.meeting_target)
            self._set_target_combo(self.local_lang_combo, config.language.local_target)
            legacy_enabled = bool(getattr(config.runtime, "translation_enabled", True))
            self.remote_translation_enabled_check.setChecked(
                bool(getattr(config.runtime, "remote_translation_enabled", legacy_enabled))
            )
            self.local_translation_enabled_check.setChecked(
                bool(getattr(config.runtime, "local_translation_enabled", legacy_enabled))
            )
            self._select_combo_data(
                self.tts_output_mode_combo,
                str(getattr(config.runtime, "tts_output_mode", "subtitle_only") or "subtitle_only"),
            )
        finally:
            self._suspend_notify = False
        self._update_source_language_controls()
        self._refresh_panel_labels()
        self.update_translation_voice_labels(config)

    def update_config(self, config: AppConfig) -> None:
        config.direction.mode = "bidirectional"
        config.language.meeting_target = self._get_target_language(self.remote_lang_combo, default="zh-TW")
        config.language.local_target = self._get_target_language(self.local_lang_combo, default="en")
        config.runtime.remote_translation_enabled = self.translation_enabled("remote")
        config.runtime.local_translation_enabled = self.translation_enabled("local")
        config.runtime.translation_enabled = self.translation_enabled()
        config.runtime.asr_language_mode = "auto"
        config.runtime.tts_output_mode = self.selected_tts_output_mode()
        tts_enabled = self.selected_tts_output_mode() == "tts"
        config.runtime.remote_tts_enabled = tts_enabled
        config.runtime.local_tts_enabled = tts_enabled

    def selected_mode(self) -> str:
        return "bidirectional"

    def selected_asr_language_mode(self) -> str:
        return "auto"

    def selected_tts_output_mode(self) -> str:
        value = self.tts_output_mode_combo.currentData()
        mode = str(value) if value else "subtitle_only"
        return mode if mode in {"tts", "subtitle_only", "passthrough"} else "subtitle_only"

    def translation_enabled(self, source: str | None = None) -> bool:
        if source == "remote":
            return self.remote_translation_enabled_check.isChecked()
        if source == "local":
            return self.local_translation_enabled_check.isChecked()
        return self.remote_translation_enabled_check.isChecked() or self.local_translation_enabled_check.isChecked()

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def set_direction_controls_enabled(self, enabled: bool) -> None:
        self._direction_controls_enabled = bool(enabled)
        for widget in (
            self.remote_translation_enabled_check,
            self.local_translation_enabled_check,
            self.tts_output_mode_combo,
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
        final_lines = [line for line in lines if not line.startswith("[partial]")]
        partial_line: str | None = lines[-1] if lines and lines[-1].startswith("[partial]") else None

        committed = self._editor_committed[editor]
        had_partial = self._editor_has_partial[editor]

        new_finals = final_lines[len(committed):]
        if new_finals:
            if had_partial:
                self._remove_editor_last_line(editor)
                had_partial = False
            for line in new_finals:
                self._append_editor_line(editor, line)
            self._editor_committed[editor] = final_lines
            self._editor_has_partial[editor] = False

        if partial_line is not None:
            if had_partial:
                self._remove_editor_last_line(editor)
            self._append_editor_line(editor, partial_line)
            self._editor_has_partial[editor] = True

        scrollbar = editor.verticalScrollBar()
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

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
        tooltip = "來源語言由 ASR 自動偵測，這裡只調整翻譯目標語言。"
        self.remote_lang_combo.setEnabled(enabled)
        self.local_lang_combo.setEnabled(enabled)
        self.remote_lang_combo.setToolTip(tooltip)
        self.local_lang_combo.setToolTip(tooltip)

    def _refresh_panel_labels(self) -> None:
        remote_target = self._get_target_language(self.remote_lang_combo, default="zh-TW")
        local_target = self._get_target_language(self.local_lang_combo, default="en")
        remote_detected = self._detected_asr_language.get("remote", "")
        local_detected = self._detected_asr_language.get("local", "")

        remote_source_text = f"自動偵測：{self._language_label(remote_detected)}" if remote_detected else "自動偵測"
        local_source_text = f"自動偵測：{self._language_label(local_detected)}" if local_detected else "自動偵測"

        self.remote_original_label.setText(f"遠端原文（{remote_source_text} ASR）")
        self.local_original_label.setText(f"本地原文（{local_source_text} ASR）")

        if self.translation_enabled("remote"):
            self.remote_translated_label.setText(f"遠端翻譯（{self._language_label(remote_target)} LLM）")
        else:
            self.remote_translated_label.setText("遠端輸出（ASR 原文）")

        if self.translation_enabled("local"):
            self.local_translated_label.setText(f"本地翻譯（{self._language_label(local_target)} LLM）")
        else:
            self.local_translated_label.setText("本地輸出（ASR 原文）")

    def update_translation_voice_labels(self, config: AppConfig) -> None:
        self._refresh_panel_labels()
        if self.selected_tts_output_mode() != "tts":
            return
        remote_target = self._get_target_language(self.remote_lang_combo, default="zh-TW")
        local_target = self._get_target_language(self.local_lang_combo, default="en")
        remote_voice = self._display_voice_label(resolve_edge_voice_for_target(config, remote_target))
        local_voice = self._display_voice_label(resolve_edge_voice_for_target(config, local_target))

        if self.translation_enabled("remote"):
            self.remote_translated_label.setText(
                f"遠端翻譯（{self._language_label(remote_target)} LLM / Edge: {remote_voice}）"
            )
        else:
            self.remote_translated_label.setText(f"遠端輸出（ASR 原文 / Edge: {remote_voice}）")

        if self.translation_enabled("local"):
            self.local_translated_label.setText(
                f"本地翻譯（{self._language_label(local_target)} LLM / Edge: {local_voice}）"
            )
        else:
            self.local_translated_label.setText(f"本地輸出（ASR 原文 / Edge: {local_voice}）")

    def _make_panel_header(
        self,
        title_label: QLabel,
        status_label: QLabel,
        export_btn: QPushButton | None = None,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(title_label)
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
        combo.setMaxVisibleItems(max(1, combo.count()))
        view = combo.view()
        if view is None:
            return
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @staticmethod
    def _display_voice_label(voice_name: str) -> str:
        value = (voice_name or "").strip()
        return value or "未設定"
