from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class QuickStartPage(QWidget):
    def __init__(self, on_apply_banana_preset: Callable[[], None]) -> None:
        super().__init__()
        self._on_apply_banana_preset = on_apply_banana_preset

        layout = QVBoxLayout(self)
        title = QLabel("Banana-only 快速開始")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        tips = QLabel(
            "1. 會議軟體 Speaker: VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)\n"
            "2. 會議軟體 Microphone: VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)\n"
            "3. 在音訊路由頁確認 remote/local 四個邏輯通道"
        )
        tips.setWordWrap(True)

        self.detected_label = QLabel("偵測到 VoiceMeeter 裝置: 0")
        self.detected_label.setWordWrap(True)

        apply_btn = QPushButton("套用 Banana 預設到音訊路由")
        apply_btn.clicked.connect(self._on_apply_banana_preset)

        layout.addWidget(title)
        layout.addWidget(tips)
        layout.addWidget(self.detected_label)
        layout.addWidget(apply_btn)
        layout.addStretch(1)

    def set_detected_voicemeeter_devices(self, device_names: list[str]) -> None:
        if not device_names:
            self.detected_label.setText("偵測到 VoiceMeeter 裝置: 0")
            return
        lines = "\n".join(f"- {name}" for name in device_names)
        self.detected_label.setText(f"偵測到 VoiceMeeter 裝置: {len(device_names)}\n{lines}")
