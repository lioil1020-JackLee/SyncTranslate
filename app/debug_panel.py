from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout, QWidget, QSizePolicy


class DebugPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.runtime_details = QTextEdit()
        self.runtime_details.setReadOnly(True)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.runtime_details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.error_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        runtime_group = QGroupBox("Runtime")
        runtime_layout = QVBoxLayout(runtime_group)
        runtime_layout.setContentsMargins(8, 8, 8, 8)
        runtime_layout.addWidget(self.runtime_details, 1)

        errors_group = QGroupBox("最近錯誤")
        errors_layout = QVBoxLayout(errors_group)
        errors_layout.setContentsMargins(8, 8, 8, 8)
        errors_layout.addWidget(self.error_log, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(runtime_group, 1)
        layout.addWidget(errors_group, 1)

    def update_runtime_stats(self, text: str) -> None:
        self.runtime_details.setPlainText(text)

    def update_recent_errors(self, errors: list[str]) -> None:
        self.error_log.setPlainText("\n".join(errors[-20:]))
