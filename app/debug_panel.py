from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from app.router import RouteCheckResult


class DebugPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel("Route check: -")
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.details)
        layout.addWidget(QLabel("最近錯誤"))
        layout.addWidget(self.error_log)

    def update_route_result(self, result: RouteCheckResult) -> None:
        self.status_label.setText(f"Route check: {result.summary}")
        if not result.issues:
            self.details.setPlainText("OK: 沒有發現路由衝突。")
            return

        lines = [f"[{issue.level}] {issue.field}: {issue.message}" for issue in result.issues]
        self.details.setPlainText("\n".join(lines))

    def update_recent_errors(self, errors: list[str]) -> None:
        self.error_log.setPlainText("\n".join(errors[-20:]))
