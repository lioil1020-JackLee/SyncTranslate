from __future__ import annotations

from typing import Callable

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

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

from app.schemas import AppConfig
from app.tts_language_resolver import resolve_edge_voice_for_target

_LANG_CHOICES: list[tuple[str, str, str]] = [
    ("\u82f1\u7ffb\u4e2d", "en", "zh-TW"),
    ("\u4e2d\u7ffb\u82f1", "zh-TW", "en"),
]

_LANG_LABELS: dict[str, str] = {
    "en": "英文",
    "zh": "中文",
    "zh-tw": "中文",
    "ja": "日文",
}


def _lang_key(source: str, target: str) -> str:
    return f"{source}|{target}"


class LiveCaptionPage(QWidget):
    def __init__(
        self,
        on_clear_clicked: Callable[[], None] | None = None,
        on_start_clicked: Callable[[], None] | None = None,
        on_test_local_tts_clicked: Callable[[], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
        on_remote_tts_changed: Callable[[bool], None] | None = None,
        on_local_tts_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_clear_clicked = on_clear_clicked
        self._on_start_clicked = on_start_clicked
        self._on_test_local_tts_clicked = on_test_local_tts_clicked
        self._on_settings_changed = on_settings_changed
        self._on_remote_tts_changed = on_remote_tts_changed
        self._on_local_tts_changed = on_local_tts_changed
        self._suspend_notify = False

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("遠端 -> 本地", "meeting_to_local")
        self.mode_combo.addItem("本地 -> 遠端", "local_to_meeting")
        self.mode_combo.addItem("雙向模式", "bidirectional")
        self._configure_combo_popup(self.mode_combo)
        self.remote_lang_combo = QComboBox()
        self.local_lang_combo = QComboBox()
        for label, source, target in _LANG_CHOICES:
            key = _lang_key(source, target)
            self.remote_lang_combo.addItem(label, key)
            self.local_lang_combo.addItem(label, key)
        self._configure_combo_popup(self.remote_lang_combo)
        self._configure_combo_popup(self.local_lang_combo)

        self.start_btn = QPushButton("\u958b\u59cb")
        self.start_btn.clicked.connect(self._handle_start_clicked)
        self.clear_btn = QPushButton("\u6e05\u7a7a\u5b57\u5e55")
        self.clear_btn.clicked.connect(self._handle_clear_clicked)
        self.test_local_tts_btn = QPushButton("測試本地輸出TTS")
        self.test_local_tts_btn.clicked.connect(self._handle_test_local_tts_clicked)

        self.remote_original = QTextEdit()
        self.remote_translated = QTextEdit()
        self.local_original = QTextEdit()
        self.local_translated = QTextEdit()

        # Export buttons — one per panel
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

        # TTS playback checkboxes — one per translated panel (default: disabled)
        self.remote_tts_enabled_check = QCheckBox("播放")
        self.remote_tts_enabled_check.setChecked(False)
        self.remote_tts_enabled_check.setToolTip("勾選時播放右上角翻譯語音；取消勾選時，遠端輸入語音會直送本地輸出")
        self.remote_tts_enabled_check.toggled.connect(self._handle_remote_tts_changed)
        self.local_tts_enabled_check = QCheckBox("播放")
        self.local_tts_enabled_check.setChecked(False)
        self.local_tts_enabled_check.setToolTip("勾選時播放右下角翻譯語音；取消勾選時，本地輸入語音會直送遠端輸出")
        self.local_tts_enabled_check.toggled.connect(self._handle_local_tts_changed)

        # Per-editor append state (for persistent history without full setPlainText rebuilds)
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

        self.remote_original_label = QLabel("\u9060\u7aef\u539f\u6587")
        self.remote_translated_label = QLabel("\u9060\u7aef\u7ffb\u8b6f")
        self.local_original_label = QLabel("\u672c\u5730\u539f\u6587")
        self.local_translated_label = QLabel("\u672c\u5730\u7ffb\u8b6f")
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

        for combo in (self.mode_combo, self.remote_lang_combo, self.local_lang_combo):
            combo.currentIndexChanged.connect(self._notify_settings_changed)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("模式"))
        top_row.addWidget(self.mode_combo, 0)
        top_row.addSpacing(8)
        top_row.addWidget(QLabel("\u9060\u7aef\u8a9e\u8a00\u65b9\u5411"))
        top_row.addWidget(self.remote_lang_combo, 0)
        top_row.addSpacing(8)
        top_row.addWidget(QLabel("\u672c\u5730\u8a9e\u8a00\u65b9\u5411"))
        top_row.addWidget(self.local_lang_combo, 0)
        top_row.addWidget(self.test_local_tts_btn, 0)
        top_row.addStretch(1)
        top_row.addWidget(self.clear_btn, 0)
        top_row.addWidget(self.start_btn, 0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.addLayout(self._make_panel_header(self.remote_original_label, self.remote_original_status, self.export_remote_original_btn), 0, 0)
        grid.addLayout(self._make_panel_header(self.remote_translated_label, self.remote_translated_status, self.export_remote_translated_btn, self.remote_tts_enabled_check), 0, 1)
        grid.addWidget(self.remote_original, 1, 0)
        grid.addWidget(self.remote_translated, 1, 1)
        grid.addLayout(self._make_panel_header(self.local_original_label, self.local_original_status, self.export_local_original_btn), 2, 0)
        grid.addLayout(self._make_panel_header(self.local_translated_label, self.local_translated_status, self.export_local_translated_btn, self.local_tts_enabled_check), 2, 1)
        grid.addWidget(self.local_original, 3, 0)
        grid.addWidget(self.local_translated, 3, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(top_row)
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
            self._set_mode_combo(config.direction.mode)
            self._set_lang_combo(self.remote_lang_combo, config.language.meeting_source, config.language.meeting_target)
            self._set_lang_combo(self.local_lang_combo, config.language.local_source, config.language.local_target)
            self.remote_tts_enabled_check.setChecked(bool(getattr(config.runtime, "remote_tts_enabled", False)))
            self.local_tts_enabled_check.setChecked(bool(getattr(config.runtime, "local_tts_enabled", False)))
        finally:
            self._suspend_notify = False
        self._refresh_panel_labels()
        self.update_translation_voice_labels(config)

    def update_config(self, config: AppConfig) -> None:
        config.direction.mode = self.selected_mode()
        r_source, r_target = self._get_lang_pair(self.remote_lang_combo, default=("en", "zh-TW"))
        l_source, l_target = self._get_lang_pair(self.local_lang_combo, default=("zh-TW", "en"))
        config.language.meeting_source = r_source
        config.language.meeting_target = r_target
        config.language.local_source = l_source
        config.language.local_target = l_target
        config.runtime.remote_tts_enabled = self.remote_tts_enabled_check.isChecked()
        config.runtime.local_tts_enabled = self.local_tts_enabled_check.isChecked()

    def selected_mode(self) -> str:
        value = self.mode_combo.currentData()
        return str(value) if value else "meeting_to_local"

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def set_direction_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.mode_combo, self.remote_lang_combo, self.local_lang_combo):
            widget.setEnabled(enabled)

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

    def _handle_start_clicked(self) -> None:
        if self._on_start_clicked:
            self._on_start_clicked()

    def _notify_settings_changed(self, *_args) -> None:
        if self._suspend_notify:
            return
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

    def _handle_remote_tts_changed(self, enabled: bool) -> None:
        if self._on_remote_tts_changed:
            self._on_remote_tts_changed(enabled)

    def _handle_local_tts_changed(self, enabled: bool) -> None:
        if self._on_local_tts_changed:
            self._on_local_tts_changed(enabled)

    def clear(self) -> None:
        for editor in (self.remote_original, self.remote_translated,
                       self.local_original, self.local_translated):
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
        """Append-only update: only new final lines are appended; partial line is updated in-place."""
        # Separate final lines (prefix "[final] ") and the optional trailing partial.
        final_lines = [l for l in lines if not l.startswith("[partial]")]
        partial_line: str | None = lines[-1] if lines and lines[-1].startswith("[partial]") else None

        committed = self._editor_committed[editor]
        had_partial = self._editor_has_partial[editor]

        # Append any newly finalised lines.
        new_finals = final_lines[len(committed):]
        if new_finals:
            if had_partial:
                self._remove_editor_last_line(editor)
                had_partial = False
            for line in new_finals:
                self._append_editor_line(editor, line)
            self._editor_committed[editor] = final_lines
            self._editor_has_partial[editor] = False

        # Update the partial (in-progress) line.
        if partial_line is not None:
            if had_partial:
                self._remove_editor_last_line(editor)
            self._append_editor_line(editor, partial_line)
            self._editor_has_partial[editor] = True
        elif had_partial and not new_finals:
            # Partial disappeared without a new final (silence flush); leave it as-is.
            pass

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
        cursor.deletePreviousChar()  # remove the preceding newline (no-op if at start)
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
    def _get_lang_pair(combo: QComboBox, default: tuple[str, str]) -> tuple[str, str]:
        data = combo.currentData()
        if isinstance(data, str) and "|" in data:
            source, target = data.split("|", 1)
            return source, target
        return default

    @staticmethod
    def _set_lang_combo(combo: QComboBox, source: str, target: str) -> None:
        idx = combo.findData(_lang_key(source, target))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_mode_combo(self, mode: str) -> None:
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def _refresh_panel_labels(self) -> None:
        remote_source, remote_target = self._get_lang_pair(self.remote_lang_combo, default=("en", "zh-TW"))
        local_source, local_target = self._get_lang_pair(self.local_lang_combo, default=("zh-TW", "en"))
        self.remote_original_label.setText(f"遠端原文（{self._language_label(remote_source)} ASR）")
        self.remote_translated_label.setText(f"遠端翻譯（{self._language_label(remote_target)} LLM）")
        self.local_original_label.setText(f"本地原文（{self._language_label(local_source)} ASR）")
        self.local_translated_label.setText(f"本地翻譯（{self._language_label(local_target)} LLM）")

    def update_translation_voice_labels(self, config: AppConfig) -> None:
        remote_target = self._get_lang_pair(self.remote_lang_combo, default=("en", "zh-TW"))[1]
        local_target = self._get_lang_pair(self.local_lang_combo, default=("zh-TW", "en"))[1]
        remote_voice = self._display_voice_label(resolve_edge_voice_for_target(config, remote_target))
        local_voice = self._display_voice_label(resolve_edge_voice_for_target(config, local_target))
        self.remote_translated_label.setText(
            f"遠端翻譯（{self._language_label(remote_target)} LLM / Edge: {remote_voice}）"
        )
        self.local_translated_label.setText(
            f"本地翻譯（{self._language_label(local_target)} LLM / Edge: {local_voice}）"
        )

    def _make_panel_header(
        self,
        title_label: QLabel,
        status_label: QLabel,
        export_btn: QPushButton | None = None,
        tts_check: QCheckBox | None = None,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(title_label)
        row.addStretch(1)
        row.addWidget(status_label)
        if tts_check is not None:
            row.addWidget(tts_check)
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
