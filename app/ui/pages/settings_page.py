from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget


class SettingsPage(QWidget):
    def __init__(
        self,
        audio_routing_page: QWidget,
        local_ai_page: QWidget,
        diagnostics_page: QWidget,
    ) -> None:
        super().__init__()
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(8)
        content_layout.addWidget(
            self._build_section(
                "音訊裝置",
                "設定會議端與本地端的輸入與輸出裝置，使用固定雙向音源線。",
                audio_routing_page,
            )
        )
        content_layout.addWidget(
            self._build_section(
                "翻譯與輸出",
                "集中放置模型、翻譯目標語言、輸出模式與進階設定。",
                local_ai_page,
            )
        )
        content_layout.addWidget(
            self._build_section(
                "診斷摘要",
                "只顯示目前 ASR / LLM / TTS 的健康檢查結果摘要。",
                diagnostics_page,
            )
        )
        content_layout.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._scroll)

    @staticmethod
    def _build_section(title: str, description: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        summary = QLabel(description)
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #b7bdc6;")
        layout.addWidget(summary)
        layout.addWidget(widget)
        return group
