from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class IoControlPage(QWidget):
    def __init__(self, audio_routing_page: QWidget, diagnostics_page: QWidget) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(audio_routing_page, 0)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider, 0)

        layout.addWidget(QLabel("執行診斷"))
        layout.addWidget(diagnostics_page, 1)
