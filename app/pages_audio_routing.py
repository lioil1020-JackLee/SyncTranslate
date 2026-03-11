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
        on_start: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_apply_banana_preset = on_apply_banana_preset
        self._on_save = on_save
        self._on_reload = on_reload
        self._on_route_changed = on_route_changed
        self._on_start = on_start
        self._input_devices: list[DeviceInfo] = []
        self._output_devices: list[DeviceInfo] = []

        self.remote_in_hostapi_combo = QComboBox()
        self.remote_in_combo = QComboBox()
        self.local_mic_in_hostapi_combo = QComboBox()
        self.local_mic_in_combo = QComboBox()
        self.local_tts_out_hostapi_combo = QComboBox()
        self.local_tts_out_combo = QComboBox()
        self.meeting_tts_out_hostapi_combo = QComboBox()
        self.meeting_tts_out_combo = QComboBox()
        self.session_mode_combo = QComboBox()
        self.start_btn = QPushButton("開始")
        self.session_mode_combo.addItem("只收聽翻譯", "remote_only")
        self.session_mode_combo.addItem("只發話翻譯", "local_only")
        self.session_mode_combo.addItem("雙向模式", "bidirectional")

        for combo in (
            self.remote_in_hostapi_combo,
            self.remote_in_combo,
            self.local_mic_in_hostapi_combo,
            self.local_mic_in_combo,
            self.local_tts_out_hostapi_combo,
            self.local_tts_out_combo,
            self.meeting_tts_out_hostapi_combo,
            self.meeting_tts_out_combo,
            self.session_mode_combo,
        ):
            combo.currentIndexChanged.connect(self._on_route_changed)

        self.remote_in_hostapi_combo.currentIndexChanged.connect(self._refresh_remote_in_combo)
        self.local_mic_in_hostapi_combo.currentIndexChanged.connect(self._refresh_local_mic_combo)
        self.local_tts_out_hostapi_combo.currentIndexChanged.connect(self._refresh_local_tts_combo)
        self.meeting_tts_out_hostapi_combo.currentIndexChanged.connect(self._refresh_meeting_tts_combo)
        self.start_btn.clicked.connect(self._on_start)

        self.status_label = QLabel("音訊路由設定")

        form = QFormLayout()
        form.addRow(
            "會議語音輸入",
            self._build_route_selector_row(self.remote_in_hostapi_combo, self.remote_in_combo),
        )
        form.addRow(
            "本機麥克風輸入",
            self._build_route_selector_row(self.local_mic_in_hostapi_combo, self.local_mic_in_combo),
        )
        form.addRow(
            "本機收聽輸出",
            self._build_route_selector_row(self.local_tts_out_hostapi_combo, self.local_tts_out_combo),
        )
        form.addRow(
            "會議送出輸出",
            self._build_route_selector_row(self.meeting_tts_out_hostapi_combo, self.meeting_tts_out_combo),
        )
        form.addRow("模式", self.session_mode_combo)

        button_row = QHBoxLayout()
        apply_btn = QPushButton("套用 Banana 預設")
        save_btn = QPushButton("儲存設定")
        reload_btn = QPushButton("重新載入")
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
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)
        current = self.selected_audio_routes()
        self._fill_hostapi_combo(self.remote_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.local_mic_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.local_tts_out_hostapi_combo, self._output_devices)
        self._fill_hostapi_combo(self.meeting_tts_out_hostapi_combo, self._output_devices)
        self._apply_selector_to_route(self.remote_in_hostapi_combo, self.remote_in_combo, self._input_devices, current.remote_in)
        self._apply_selector_to_route(
            self.local_mic_in_hostapi_combo,
            self.local_mic_in_combo,
            self._input_devices,
            current.local_mic_in,
        )
        self._apply_selector_to_route(
            self.local_tts_out_hostapi_combo,
            self.local_tts_out_combo,
            self._output_devices,
            current.local_tts_out,
        )
        self._apply_selector_to_route(
            self.meeting_tts_out_hostapi_combo,
            self.meeting_tts_out_combo,
            self._output_devices,
            current.meeting_tts_out,
        )

    def apply_config(self, config: AppConfig) -> None:
        self._apply_selector_to_route(self.remote_in_hostapi_combo, self.remote_in_combo, self._input_devices, config.audio.remote_in)
        self._apply_selector_to_route(
            self.local_mic_in_hostapi_combo,
            self.local_mic_in_combo,
            self._input_devices,
            config.audio.local_mic_in,
        )
        self._apply_selector_to_route(
            self.local_tts_out_hostapi_combo,
            self.local_tts_out_combo,
            self._output_devices,
            config.audio.local_tts_out,
        )
        self._apply_selector_to_route(
            self.meeting_tts_out_hostapi_combo,
            self.meeting_tts_out_combo,
            self._output_devices,
            config.audio.meeting_tts_out,
        )
        self._select_by_mode(self.session_mode_combo, config.session_mode)

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            remote_in=self._selected_selector(self.remote_in_combo),
            local_mic_in=self._selected_selector(self.local_mic_in_combo),
            local_tts_out=self._selected_selector(self.local_tts_out_combo),
            meeting_tts_out=self._selected_selector(self.meeting_tts_out_combo),
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

    def _refresh_remote_in_combo(self) -> None:
        self._refill_device_combo(self.remote_in_combo, self._input_devices, self.remote_in_hostapi_combo.currentData())

    def _refresh_local_mic_combo(self) -> None:
        self._refill_device_combo(self.local_mic_in_combo, self._input_devices, self.local_mic_in_hostapi_combo.currentData())

    def _refresh_local_tts_combo(self) -> None:
        self._refill_device_combo(self.local_tts_out_combo, self._output_devices, self.local_tts_out_hostapi_combo.currentData())

    def _refresh_meeting_tts_combo(self) -> None:
        self._refill_device_combo(
            self.meeting_tts_out_combo,
            self._output_devices,
            self.meeting_tts_out_hostapi_combo.currentData(),
        )

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
            label = (
                f"[{device.hostapi_label}] [{device.index}] {device.name} "
                f"(ch in/out: {device.max_input_channels}/{device.max_output_channels}, "
                f"sr: {device.default_samplerate:.0f})"
            )
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
        hostapi_combo.setToolTip("音訊介面")
        device_combo.setToolTip("音訊裝置")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(hostapi_combo, 1)
        row.addWidget(device_combo, 2)
        return container
