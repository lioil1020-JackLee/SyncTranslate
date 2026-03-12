from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class DiagnosticsPage(QWidget):
    def __init__(
        self,
        on_health_check: Callable[[bool], None],
        on_test_meeting_tts: Callable[[], None],
        on_test_speaker_tts: Callable[[], None],
        on_export_diagnostics: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_health_check = on_health_check
        self._on_test_meeting_tts = on_test_meeting_tts
        self._on_test_speaker_tts = on_test_speaker_tts
        self._on_export_diagnostics = on_export_diagnostics

        title = QLabel("\u672c\u5730\u57f7\u884c\u8a3a\u65b7")
        self.health_label = QLabel("health: -")
        self.details = QTextEdit()
        self.details.setReadOnly(True)

        buttons = QHBoxLayout()
        check_btn = QPushButton("\u5065\u5eb7\u6aa2\u67e5")
        warmup_btn = QPushButton("\u9810\u71b1 + \u6aa2\u67e5")
        test_meeting_btn = QPushButton("\u6e2c\u8a66\u6703\u8b70\u8f38\u51faTTS")
        test_speaker_btn = QPushButton("\u6e2c\u8a66\u5587\u53ed\u8f38\u51faTTS")
        export_btn = QPushButton("\u532f\u51fa\u8a3a\u65b7\u8cc7\u8a0a")
        check_btn.clicked.connect(lambda: self._on_health_check(False))
        warmup_btn.clicked.connect(lambda: self._on_health_check(True))
        test_meeting_btn.clicked.connect(self._on_test_meeting_tts)
        test_speaker_btn.clicked.connect(self._on_test_speaker_tts)
        export_btn.clicked.connect(self._on_export_diagnostics)
        buttons.addWidget(check_btn)
        buttons.addWidget(warmup_btn)
        buttons.addWidget(test_meeting_btn)
        buttons.addWidget(test_speaker_btn)
        buttons.addWidget(export_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.health_label)
        layout.addLayout(buttons)
        layout.addWidget(self.details)
        layout.addStretch(1)

    def set_health_summary(self, text: str) -> None:
        self.health_label.setText(text)

    def set_health_details(self, text: str) -> None:
        self.details.setPlainText(text)
