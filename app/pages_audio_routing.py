from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.schemas import AppConfig, AudioRouteConfig, DeviceInfo


class AudioRoutingPage(QWidget):
    def __init__(
        self,
        on_apply_banana_preset: Callable[[], None],
        on_save: Callable[[], None],
        on_reload: Callable[[], None],
        on_route_changed: Callable[[], None],
        on_start: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_apply_banana_preset = on_apply_banana_preset
        self._on_save = on_save
        self._on_reload = on_reload
        self._on_route_changed = on_route_changed
        self._on_start = on_start

        self.remote_in_combo = QComboBox()
        self.local_mic_in_combo = QComboBox()
        self.local_tts_out_combo = QComboBox()
        self.meeting_tts_out_combo = QComboBox()
        self.session_mode_combo = QComboBox()
        self.start_btn = QPushButton("開始")
        self.session_mode_combo.addItem("只收聽翻譯", "remote_only")
        self.session_mode_combo.addItem("只發話翻譯", "local_only")
        self.session_mode_combo.addItem("雙向模式", "bidirectional")

        self.remote_in_combo.currentIndexChanged.connect(self._on_route_changed)
        self.local_mic_in_combo.currentIndexChanged.connect(self._on_route_changed)
        self.local_tts_out_combo.currentIndexChanged.connect(self._on_route_changed)
        self.meeting_tts_out_combo.currentIndexChanged.connect(self._on_route_changed)
        self.session_mode_combo.currentIndexChanged.connect(self._on_route_changed)
        self.start_btn.clicked.connect(self._on_start)

        self.status_label = QLabel("尚未儲存")

        form = QFormLayout()
        form.addRow("remote_in (輸入)", self.remote_in_combo)
        form.addRow("local_mic_in (輸入)", self.local_mic_in_combo)
        form.addRow("local_tts_out (輸出)", self.local_tts_out_combo)
        form.addRow("meeting_tts_out (輸出)", self.meeting_tts_out_combo)
        form.addRow("模式", self.session_mode_combo)

        button_row = QHBoxLayout()
        apply_btn = QPushButton("載入 Banana 預設")
        save_btn = QPushButton("儲存設定")
        reload_btn = QPushButton("重載設定")
        apply_btn.clicked.connect(self._on_apply_banana_preset)
        save_btn.clicked.connect(self._on_save)
        reload_btn.clicked.connect(self._on_reload)
        button_row.addWidget(apply_btn)
        button_row.addWidget(save_btn)
        button_row.addWidget(reload_btn)
        button_row.addWidget(self.start_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._fill_device_combo(self.remote_in_combo, input_devices)
        self._fill_device_combo(self.local_mic_in_combo, input_devices)
        self._fill_device_combo(self.local_tts_out_combo, output_devices)
        self._fill_device_combo(self.meeting_tts_out_combo, output_devices)

    def apply_config(self, config: AppConfig) -> None:
        self._select_by_name(self.remote_in_combo, config.audio.remote_in)
        self._select_by_name(self.local_mic_in_combo, config.audio.local_mic_in)
        self._select_by_name(self.local_tts_out_combo, config.audio.local_tts_out)
        self._select_by_name(self.meeting_tts_out_combo, config.audio.meeting_tts_out)
        self._select_by_mode(self.session_mode_combo, config.session_mode)

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            remote_in=self._selected_name(self.remote_in_combo),
            local_mic_in=self._selected_name(self.local_mic_in_combo),
            local_tts_out=self._selected_name(self.local_tts_out_combo),
            meeting_tts_out=self._selected_name(self.meeting_tts_out_combo),
        )

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)

    def set_start_label(self, text: str) -> None:
        self.start_btn.setText(text)

    def selected_mode(self) -> str:
        value = self.session_mode_combo.currentData()
        return str(value) if value else "remote_only"

    @staticmethod
    def _fill_device_combo(combo: QComboBox, devices: list[DeviceInfo]) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", "")
        for device in devices:
            label = (
                f"[{device.index}] {device.name} "
                f"(ch in/out: {device.max_input_channels}/{device.max_output_channels}, "
                f"sr: {device.default_samplerate:.0f})"
            )
            combo.addItem(label, device.name)
        combo.blockSignals(False)

    @staticmethod
    def _selected_name(combo: QComboBox) -> str:
        value = combo.currentData()
        return str(value) if value else ""

    @staticmethod
    def _select_by_name(combo: QComboBox, name: str) -> None:
        index = combo.findData(name)
        if index >= 0:
            combo.setCurrentIndex(index)
            return

        name_lower = name.lower()
        for idx in range(combo.count()):
            value = combo.itemData(idx)
            if isinstance(value, str) and value.lower() == name_lower:
                combo.setCurrentIndex(idx)
                return

        combo.setCurrentIndex(0)

    @staticmethod
    def _select_by_mode(combo: QComboBox, mode: str) -> None:
        index = combo.findData(mode)
        if index >= 0:
            combo.setCurrentIndex(index)
