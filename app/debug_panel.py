from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout, QWidget, QSizePolicy


class DebugPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_runtime_text = ""
        self._last_error_text = ""
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
        if text == self._last_runtime_text:
            return
        self._last_runtime_text = text
        self.runtime_details.setPlainText(text)

    def update_recent_errors(self, errors: list[str]) -> None:
        new_text = "\n".join(errors[-20:])
        if new_text == self._last_error_text:
            return

        scrollbar = self.error_log.verticalScrollBar()
        prev_value = scrollbar.value()
        was_at_bottom = prev_value >= max(0, scrollbar.maximum() - 2)

        self._last_error_text = new_text
        self.error_log.setPlainText(new_text)

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(prev_value, scrollbar.maximum()))
