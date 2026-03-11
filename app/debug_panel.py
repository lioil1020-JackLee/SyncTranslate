from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from app.router import RouteCheckResult


class DebugPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel("Route check: -")
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.runtime_details = QTextEdit()
        self.runtime_details.setReadOnly(True)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.details)
        layout.addWidget(QLabel("Runtime"))
        layout.addWidget(self.runtime_details)
        layout.addWidget(QLabel("最近錯誤"))
        layout.addWidget(self.error_log)

    def update_route_result(self, result: RouteCheckResult) -> None:
        self.status_label.setText(f"Route check: {result.summary}")
        if not result.issues:
            self.details.setPlainText("OK: 音訊路由設定完整")
            return

        lines = [f"[{issue.level}] {issue.field}: {issue.message}" for issue in result.issues]
        self.details.setPlainText("\n".join(lines))

    def update_runtime_stats(self, text: str) -> None:
        self.runtime_details.setPlainText(text)

    def update_recent_errors(self, errors: list[str]) -> None:
        self.error_log.setPlainText("\n".join(errors[-20:]))
