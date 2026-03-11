from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.audio_device_selection import canonical_device_name, encode_device_selector, hostapi_sort_key, parse_device_selector
from app.schemas import AppConfig, AudioRouteConfig, DeviceInfo


class AudioRoutingPage(QWidget):
    def __init__(
        self,
        on_apply_banana_preset: Callable[[], None],
        on_save: Callable[[], None],
        on_reload: Callable[[], None],
        on_route_changed: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_apply_banana_preset = on_apply_banana_preset
        self._on_save = on_save
        self._on_reload = on_reload
        self._on_route_changed = on_route_changed
        self._input_devices: list[DeviceInfo] = []
        self._output_devices: list[DeviceInfo] = []

        self.meeting_in_hostapi_combo = QComboBox()
        self.meeting_in_combo = QComboBox()
        self.microphone_in_hostapi_combo = QComboBox()
        self.microphone_in_combo = QComboBox()
        self.speaker_out_hostapi_combo = QComboBox()
        self.speaker_out_combo = QComboBox()
        self.meeting_out_hostapi_combo = QComboBox()
        self.meeting_out_combo = QComboBox()
        self.direction_mode_combo = QComboBox()

        self.direction_mode_combo.addItem("\u9060\u7aef -> \u672c\u5730", "meeting_to_local")
        self.direction_mode_combo.addItem("\u672c\u5730 -> \u9060\u7aef", "local_to_meeting")
        self.direction_mode_combo.addItem("\u96d9\u5411\u6a21\u5f0f", "bidirectional")

        for combo in (
            self.meeting_in_hostapi_combo,
            self.meeting_in_combo,
            self.microphone_in_hostapi_combo,
            self.microphone_in_combo,
            self.speaker_out_hostapi_combo,
            self.speaker_out_combo,
            self.meeting_out_hostapi_combo,
            self.meeting_out_combo,
            self.direction_mode_combo,
        ):
            combo.currentIndexChanged.connect(self._on_route_changed)

        self.meeting_in_hostapi_combo.currentIndexChanged.connect(self._refresh_meeting_in_combo)
        self.microphone_in_hostapi_combo.currentIndexChanged.connect(self._refresh_microphone_in_combo)
        self.speaker_out_hostapi_combo.currentIndexChanged.connect(self._refresh_speaker_out_combo)
        self.meeting_out_hostapi_combo.currentIndexChanged.connect(self._refresh_meeting_out_combo)
        self.status_label = QLabel("\u97f3\u8a0a\u8def\u7531\u8a2d\u5b9a")

        form = QFormLayout()
        form.addRow("\u9060\u7aef\u8f38\u5165", self._build_route_selector_row(self.meeting_in_hostapi_combo, self.meeting_in_combo))
        form.addRow(
            "\u9ea5\u514b\u98a8\u8f38\u5165",
            self._build_route_selector_row(self.microphone_in_hostapi_combo, self.microphone_in_combo),
        )
        form.addRow("\u55c7\u53ed\u8f38\u51fa", self._build_route_selector_row(self.speaker_out_hostapi_combo, self.speaker_out_combo))
        form.addRow("\u9060\u7aef\u8f38\u51fa", self._build_route_selector_row(self.meeting_out_hostapi_combo, self.meeting_out_combo))
        form.addRow("\u6a21\u5f0f", self.direction_mode_combo)

        button_row = QHBoxLayout()
        apply_btn = QPushButton("\u5957\u7528 Banana \u9810\u8a2d")
        save_btn = QPushButton("\u5132\u5b58\u8a2d\u5b9a")
        reload_btn = QPushButton("\u91cd\u65b0\u8f09\u5165")
        apply_btn.clicked.connect(self._on_apply_banana_preset)
        save_btn.clicked.connect(self._on_save)
        reload_btn.clicked.connect(self._on_reload)
        button_row.addWidget(apply_btn)
        button_row.addWidget(save_btn)
        button_row.addWidget(reload_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)
        current = self.selected_audio_routes()
        self._fill_hostapi_combo(self.meeting_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.microphone_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.speaker_out_hostapi_combo, self._output_devices)
        self._fill_hostapi_combo(self.meeting_out_hostapi_combo, self._output_devices)
        self._apply_selector_to_route(self.meeting_in_hostapi_combo, self.meeting_in_combo, self._input_devices, current.meeting_in)
        self._apply_selector_to_route(
            self.microphone_in_hostapi_combo,
            self.microphone_in_combo,
            self._input_devices,
            current.microphone_in,
        )
        self._apply_selector_to_route(
            self.speaker_out_hostapi_combo,
            self.speaker_out_combo,
            self._output_devices,
            current.speaker_out,
        )
        self._apply_selector_to_route(
            self.meeting_out_hostapi_combo,
            self.meeting_out_combo,
            self._output_devices,
            current.meeting_out,
        )

    def apply_config(self, config: AppConfig) -> None:
        self._apply_selector_to_route(self.meeting_in_hostapi_combo, self.meeting_in_combo, self._input_devices, config.audio.meeting_in)
        self._apply_selector_to_route(
            self.microphone_in_hostapi_combo,
            self.microphone_in_combo,
            self._input_devices,
            config.audio.microphone_in,
        )
        self._apply_selector_to_route(
            self.speaker_out_hostapi_combo,
            self.speaker_out_combo,
            self._output_devices,
            config.audio.speaker_out,
        )
        self._apply_selector_to_route(
            self.meeting_out_hostapi_combo,
            self.meeting_out_combo,
            self._output_devices,
            config.audio.meeting_out,
        )
        self._select_by_mode(self.direction_mode_combo, config.direction.mode)

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            meeting_in=self._selected_selector(self.meeting_in_combo),
            microphone_in=self._selected_selector(self.microphone_in_combo),
            speaker_out=self._selected_selector(self.speaker_out_combo),
            meeting_out=self._selected_selector(self.meeting_out_combo),
        )

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def selected_mode(self) -> str:
        value = self.direction_mode_combo.currentData()
        return str(value) if value else "meeting_to_local"

    def _refresh_meeting_in_combo(self) -> None:
        self._refill_device_combo(self.meeting_in_combo, self._input_devices, self.meeting_in_hostapi_combo.currentData())

    def _refresh_microphone_in_combo(self) -> None:
        self._refill_device_combo(self.microphone_in_combo, self._input_devices, self.microphone_in_hostapi_combo.currentData())

    def _refresh_speaker_out_combo(self) -> None:
        self._refill_device_combo(self.speaker_out_combo, self._output_devices, self.speaker_out_hostapi_combo.currentData())

    def _refresh_meeting_out_combo(self) -> None:
        self._refill_device_combo(self.meeting_out_combo, self._output_devices, self.meeting_out_hostapi_combo.currentData())

    def _apply_selector_to_route(
        self,
        hostapi_combo: QComboBox,
        device_combo: QComboBox,
        devices: list[DeviceInfo],
        selector: str,
    ) -> None:
        hostapi_name, device_name = parse_device_selector(selector)
        if not hostapi_name and device_name:
            target_name = canonical_device_name(device_name).lower()
            matched = next(
                (device.hostapi_name for device in devices if canonical_device_name(device.name).lower() == target_name),
                "",
            )
            hostapi_name = matched
        self._select_hostapi(hostapi_combo, hostapi_name)
        self._refill_device_combo(device_combo, devices, hostapi_name)
        self._select_device_selector(device_combo, selector or device_name)

    @staticmethod
    def _fill_hostapi_combo(combo: QComboBox, devices: list[DeviceInfo]) -> None:
        current = combo.currentData()
        hostapi_names = sorted({device.hostapi_name for device in devices if device.hostapi_name}, key=hostapi_sort_key)
        combo.blockSignals(True)
        combo.clear()
        for hostapi_name in hostapi_names:
            label = next(device.hostapi_label for device in devices if device.hostapi_name == hostapi_name)
            combo.addItem(label, hostapi_name)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)

    @staticmethod
    def _refill_device_combo(combo: QComboBox, devices: list[DeviceInfo], hostapi_name: str | None) -> None:
        current = combo.currentData()
        selected_hostapi = str(hostapi_name or "")
        filtered = [device for device in devices if not selected_hostapi or device.hostapi_name == selected_hostapi]
        combo.blockSignals(True)
        combo.clear()
        for device in filtered:
            label = f"[{device.hostapi_label}] [{device.index}] {device.name}"
            selector = encode_device_selector(hostapi_name=device.hostapi_name, device_name=device.name)
            combo.addItem(label, selector)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)

    @staticmethod
    def _selected_selector(combo: QComboBox) -> str:
        value = combo.currentData()
        return str(value) if value else ""

    @staticmethod
    def _select_hostapi(combo: QComboBox, hostapi_name: str) -> None:
        index = combo.findData(hostapi_name)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _select_device_selector(combo: QComboBox, selector: str) -> None:
        index = combo.findData(selector)
        if index >= 0:
            combo.setCurrentIndex(index)
            return

        target_name = canonical_device_name(selector)
        target_lower = target_name.lower()
        for idx in range(combo.count()):
            value = combo.itemData(idx)
            if isinstance(value, str) and canonical_device_name(value).lower() == target_lower:
                combo.setCurrentIndex(idx)
                return
        combo.setCurrentIndex(0)

    @staticmethod
    def _select_by_mode(combo: QComboBox, mode: str) -> None:
        index = combo.findData(mode)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _build_route_selector_row(hostapi_combo: QComboBox, device_combo: QComboBox) -> QWidget:
        hostapi_combo.setMinimumWidth(180)
        hostapi_combo.setToolTip("\u97f3\u8a0a\u4ecb\u9762")
        device_combo.setToolTip("\u97f3\u8a0a\u88dd\u7f6e")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(hostapi_combo, 1)
        row.addWidget(device_combo, 2)
        return container
