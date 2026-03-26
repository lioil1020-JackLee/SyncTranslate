from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.infra.audio.device_registry import (
    canonical_device_name,
    encode_device_selector,
    hostapi_sort_key,
    parse_device_selector,
)
from app.infra.config.schema import AppConfig, AudioRouteConfig, DeviceInfo


class AudioRoutingPage(QWidget):
    def __init__(
        self,
        on_route_changed,
    ) -> None:
        super().__init__()
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

        for combo in (
            self.meeting_in_hostapi_combo,
            self.meeting_in_combo,
            self.microphone_in_hostapi_combo,
            self.microphone_in_combo,
            self.speaker_out_hostapi_combo,
            self.speaker_out_combo,
            self.meeting_out_hostapi_combo,
            self.meeting_out_combo,
        ):
            self._configure_combo_popup(combo)
            combo.currentIndexChanged.connect(self._on_route_changed)

        self.meeting_in_hostapi_combo.currentIndexChanged.connect(self._refresh_meeting_in_combo)
        self.microphone_in_hostapi_combo.currentIndexChanged.connect(self._refresh_microphone_in_combo)
        self.speaker_out_hostapi_combo.currentIndexChanged.connect(self._refresh_speaker_out_combo)
        self.meeting_out_hostapi_combo.currentIndexChanged.connect(self._refresh_meeting_out_combo)

        form = QFormLayout()
        form.addRow(
            "遠端輸入",
            self._build_route_selector_row(
                self.meeting_in_hostapi_combo,
                self.meeting_in_combo,
            ),
        )
        form.addRow(
            "遠端輸出",
            self._build_route_selector_row(
                self.meeting_out_hostapi_combo,
                self.meeting_out_combo,
            ),
        )
        form.addRow(
            "本地輸入",
            self._build_route_selector_row(
                self.microphone_in_hostapi_combo,
                self.microphone_in_combo,
            ),
        )
        form.addRow(
            "本地輸出",
            self._build_route_selector_row(
                self.speaker_out_hostapi_combo,
                self.speaker_out_combo,
            ),
        )

        self.route_status_label = QLabel("請先確認四個音訊角色都已選到可用裝置。")
        self.route_status_label.setWordWrap(True)
        self.route_status_label.setStyleSheet("color: #b7bdc6;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.route_status_label)
        layout.addStretch(1)

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)
        current = self.selected_audio_routes()

        self._fill_hostapi_combo(self.meeting_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.microphone_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.speaker_out_hostapi_combo, self._output_devices)
        self._fill_hostapi_combo(self.meeting_out_hostapi_combo, self._output_devices)
        self._apply_selector_to_route(
            self.meeting_in_hostapi_combo,
            self.meeting_in_combo,
            self._input_devices,
            current.meeting_in,
        )
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
        meeting_in_selector = config.audio.meeting_in
        meeting_out_selector = config.audio.meeting_out
        self._apply_selector_to_route(
            self.meeting_in_hostapi_combo,
            self.meeting_in_combo,
            self._input_devices,
            meeting_in_selector,
        )
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
            meeting_out_selector,
        )
        config.audio.meeting_in = self._selected_selector(self.meeting_in_combo) or meeting_in_selector
        config.audio.meeting_out = self._selected_selector(self.meeting_out_combo) or meeting_out_selector
        self._force_route_levels_100(config.audio)

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            meeting_in=self._selected_selector(self.meeting_in_combo),
            microphone_in=self._selected_selector(self.microphone_in_combo),
            speaker_out=self._selected_selector(self.speaker_out_combo),
            meeting_out=self._selected_selector(self.meeting_out_combo),
            meeting_in_gain=1.0,
            microphone_in_gain=1.0,
            speaker_out_volume=1.0,
            meeting_out_volume=1.0,
        )

    def set_validation_message(self, text: str, *, is_error: bool) -> None:
        message = text.strip() or "請先確認四個音訊角色都已選到可用裝置。"
        color = "#ef4444" if is_error else "#22c55e"
        self.route_status_label.setText(message)
        self.route_status_label.setStyleSheet(f"color: {color};")

    @staticmethod
    def _force_route_levels_100(audio: AudioRouteConfig) -> None:
        audio.meeting_in_gain = 1.0
        audio.microphone_in_gain = 1.0
        audio.speaker_out_volume = 1.0
        audio.meeting_out_volume = 1.0

    def _refresh_microphone_in_combo(self) -> None:
        self._refill_device_combo(self.microphone_in_combo, self._input_devices, self.microphone_in_hostapi_combo.currentData())

    def _refresh_speaker_out_combo(self) -> None:
        self._refill_device_combo(self.speaker_out_combo, self._output_devices, self.speaker_out_hostapi_combo.currentData())

    def _refresh_meeting_in_combo(self) -> None:
        self._refill_device_combo(self.meeting_in_combo, self._input_devices, self.meeting_in_hostapi_combo.currentData())

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
        AudioRoutingPage._configure_combo_popup(combo)

    @staticmethod
    def _refill_device_combo(combo: QComboBox, devices: list[DeviceInfo], hostapi_name: str | None) -> None:
        current = combo.currentData()
        selected_hostapi = str(hostapi_name or "")
        filtered = [device for device in devices if not selected_hostapi or device.hostapi_name == selected_hostapi]
        combo.blockSignals(True)
        combo.clear()
        for device in filtered:
            label = f"[{device.index}] {device.name}"
            selector = encode_device_selector(hostapi_name=device.hostapi_name, device_name=device.name)
            combo.addItem(label, selector)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)
        AudioRoutingPage._configure_combo_popup(combo)

    @staticmethod
    def _configure_combo_popup(combo: QComboBox) -> None:
        combo.setMaxVisibleItems(max(1, combo.count()))
        view = combo.view()
        if view is None:
            return
        view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
    def _build_route_selector_row(
        hostapi_combo: QComboBox,
        device_combo: QComboBox,
    ) -> QWidget:
        hostapi_combo.setMinimumWidth(180)
        hostapi_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        hostapi_combo.setMinimumContentsLength(10)
        hostapi_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hostapi_combo.setToolTip("音訊介面")
        device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        device_combo.setMinimumContentsLength(20)
        device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        device_combo.setToolTip("音訊裝置")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(hostapi_combo, 1)
        row.addWidget(device_combo, 3)
        return container
