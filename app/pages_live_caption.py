from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from app.schemas import AppConfig

_LANG_CHOICES: list[tuple[str, str, str]] = [
    ("\u82f1\u7ffb\u4e2d", "en", "zh-TW"),
    ("\u4e2d\u7ffb\u82f1", "zh-TW", "en"),
]


def _lang_key(source: str, target: str) -> str:
    return f"{source}|{target}"


class LiveCaptionPage(QWidget):
    def __init__(
        self,
        on_clear_clicked: Callable[[], None] | None = None,
        on_start_clicked: Callable[[], None] | None = None,
        on_test_local_tts_clicked: Callable[[], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_clear_clicked = on_clear_clicked
        self._on_start_clicked = on_start_clicked
        self._on_test_local_tts_clicked = on_test_local_tts_clicked
        self._on_settings_changed = on_settings_changed
        self._suspend_notify = False

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("遠端 -> 本地", "meeting_to_local")
        self.mode_combo.addItem("本地 -> 遠端", "local_to_meeting")
        self.mode_combo.addItem("雙向模式", "bidirectional")
        self.remote_lang_combo = QComboBox()
        self.local_lang_combo = QComboBox()
        for label, source, target in _LANG_CHOICES:
            key = _lang_key(source, target)
            self.remote_lang_combo.addItem(label, key)
            self.local_lang_combo.addItem(label, key)

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
        grid.addWidget(QLabel("\u9060\u7aef\u539f\u6587"), 0, 0)
        grid.addWidget(QLabel("\u9060\u7aef\u7ffb\u8b6f"), 0, 1)
        grid.addWidget(self.remote_original, 1, 0)
        grid.addWidget(self.remote_translated, 1, 1)
        grid.addWidget(QLabel("\u672c\u5730\u539f\u6587"), 2, 0)
        grid.addWidget(QLabel("\u672c\u5730\u7ffb\u8b6f"), 2, 1)
        grid.addWidget(self.local_original, 3, 0)
        grid.addWidget(self.local_translated, 3, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(top_row)
        layout.addLayout(grid)

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._set_mode_combo(config.direction.mode)
            self._set_lang_combo(self.remote_lang_combo, config.language.meeting_source, config.language.meeting_target)
            self._set_lang_combo(self.local_lang_combo, config.language.local_source, config.language.local_target)
        finally:
            self._suspend_notify = False

    def update_config(self, config: AppConfig) -> None:
        config.direction.mode = self.selected_mode()
        r_source, r_target = self._get_lang_pair(self.remote_lang_combo, default=("en", "zh-TW"))
        l_source, l_target = self._get_lang_pair(self.local_lang_combo, default=("zh-TW", "en"))
        config.language.meeting_source = r_source
        config.language.meeting_target = r_target
        config.language.local_source = l_source
        config.language.local_target = l_target

    def selected_mode(self) -> str:
        value = self.mode_combo.currentData()
        return str(value) if value else "meeting_to_local"

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def _handle_start_clicked(self) -> None:
        if self._on_start_clicked:
            self._on_start_clicked()

    def _notify_settings_changed(self, *_args) -> None:
        if self._suspend_notify:
            return
        if self._on_settings_changed:
            self._on_settings_changed()

    def _handle_clear_clicked(self) -> None:
        self.clear()
        if self._on_clear_clicked:
            self._on_clear_clicked()

    def _handle_test_local_tts_clicked(self) -> None:
        if self._on_test_local_tts_clicked:
            self._on_test_local_tts_clicked()

    def clear(self) -> None:
        self.remote_original.clear()
        self.remote_translated.clear()
        self.local_original.clear()
        self.local_translated.clear()

    def set_remote_original_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.remote_original, lines)

    def set_remote_translated_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.remote_translated, lines)

    def set_local_original_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.local_original, lines)

    def set_local_translated_lines(self, lines: list[str]) -> None:
        self._set_editor_lines(self.local_translated, lines)

    @staticmethod
    def _set_editor_lines(editor: QTextEdit, lines: list[str]) -> None:
        padded_lines = [*lines, "", "", ""]
        editor.setPlainText("\n".join(padded_lines))
        scrollbar = editor.verticalScrollBar()
        QTimer.singleShot(0, lambda: scrollbar.setValue(scrollbar.maximum()))

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
