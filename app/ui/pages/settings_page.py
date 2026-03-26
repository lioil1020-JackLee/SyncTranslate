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
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)
        content_layout.addWidget(
            self._build_section(
                "音訊裝置",
                "設定遠端與本地的輸入/輸出裝置，包含虛擬音效卡、藍牙耳機與一般喇叭。",
                audio_routing_page,
            )
        )
        content_layout.addWidget(
            self._build_section(
                "翻譯與輸出",
                "先用快速設定完成大部分需求；只有需要時再展開進階參數。",
                local_ai_page,
            )
        )
        content_layout.addWidget(
            self._build_section(
                "儲存與診斷",
                "集中放置儲存、重載、系統檢查與診斷匯出，不再拆成額外子頁。",
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        summary = QLabel(description)
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #b7bdc6;")
        layout.addWidget(summary)
        layout.addWidget(widget)
        return group
