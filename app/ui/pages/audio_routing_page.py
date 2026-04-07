from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
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
        self._suspend_notify = False

        self.meeting_in_hostapi_combo = QComboBox()
        self.meeting_in_combo = QComboBox()
        self.microphone_in_hostapi_combo = QComboBox()
        self.microphone_in_combo = QComboBox()
        self.speaker_out_hostapi_combo = QComboBox()
        self.speaker_out_combo = QComboBox()
        self.meeting_out_hostapi_combo = QComboBox()
        self.meeting_out_combo = QComboBox()

        self.meeting_in_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.meeting_in_gain_value_label = QLabel()
        self.microphone_in_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.microphone_in_gain_value_label = QLabel()
        self.speaker_out_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.speaker_out_volume_value_label = QLabel()
        self.meeting_out_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.meeting_out_volume_value_label = QLabel()

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
            combo.currentIndexChanged.connect(self._notify_route_changed)

        for slider, value_label in (
            (self.meeting_in_gain_slider, self.meeting_in_gain_value_label),
            (self.microphone_in_gain_slider, self.microphone_in_gain_value_label),
            (self.speaker_out_volume_slider, self.speaker_out_volume_value_label),
            (self.meeting_out_volume_slider, self.meeting_out_volume_value_label),
        ):
            self._configure_gain_slider(slider)
            slider.valueChanged.connect(
                lambda value, current_label=value_label: self._update_gain_value_label(
                    current_label,
                    self._slider_value_to_gain(value),
                )
            )
            slider.valueChanged.connect(self._notify_route_changed)

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
                self.meeting_in_gain_slider,
                self.meeting_in_gain_value_label,
            ),
        )
        form.addRow(
            "遠端輸出",
            self._build_route_selector_row(
                self.meeting_out_hostapi_combo,
                self.meeting_out_combo,
                self.meeting_out_volume_slider,
                self.meeting_out_volume_value_label,
            ),
        )
        form.addRow(
            "本地輸入",
            self._build_route_selector_row(
                self.microphone_in_hostapi_combo,
                self.microphone_in_combo,
                self.microphone_in_gain_slider,
                self.microphone_in_gain_value_label,
            ),
        )
        form.addRow(
            "本地輸出",
            self._build_route_selector_row(
                self.speaker_out_hostapi_combo,
                self.speaker_out_combo,
                self.speaker_out_volume_slider,
                self.speaker_out_volume_value_label,
            ),
        )

        self.route_status_label = QLabel("請先確認四個音訊裝置都指向正確的輸入與輸出。")
        self.route_status_label.setWordWrap(True)
        self.route_status_label.setStyleSheet("color: #b7bdc6;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.route_status_label)
        layout.addStretch(1)
        self._sync_gain_labels()

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)
        current = self.selected_audio_routes()

        self._suspend_notify = True
        try:
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
        finally:
            self._suspend_notify = False

    def apply_config(self, config: AppConfig) -> None:
        meeting_in_selector = config.audio.meeting_in
        meeting_out_selector = config.audio.meeting_out

        self._suspend_notify = True
        try:
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
            self.meeting_in_gain_slider.setValue(self._gain_to_slider_value(config.audio.meeting_in_gain))
            self.microphone_in_gain_slider.setValue(self._gain_to_slider_value(config.audio.microphone_in_gain))
            self.speaker_out_volume_slider.setValue(self._gain_to_slider_value(config.audio.speaker_out_volume))
            self.meeting_out_volume_slider.setValue(self._gain_to_slider_value(config.audio.meeting_out_volume))
        finally:
            self._suspend_notify = False

        config.audio.meeting_in = self._selected_selector(self.meeting_in_combo) or meeting_in_selector
        config.audio.meeting_out = self._selected_selector(self.meeting_out_combo) or meeting_out_selector
        self._sync_gain_labels()

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            meeting_in=self._selected_selector(self.meeting_in_combo),
            microphone_in=self._selected_selector(self.microphone_in_combo),
            speaker_out=self._selected_selector(self.speaker_out_combo),
            meeting_out=self._selected_selector(self.meeting_out_combo),
            meeting_in_gain=self._slider_value_to_gain(self.meeting_in_gain_slider.value()),
            microphone_in_gain=self._slider_value_to_gain(self.microphone_in_gain_slider.value()),
            speaker_out_volume=self._slider_value_to_gain(self.speaker_out_volume_slider.value()),
            meeting_out_volume=self._slider_value_to_gain(self.meeting_out_volume_slider.value()),
        )

    def set_validation_message(self, text: str, *, is_error: bool) -> None:
        message = text.strip() or "請先確認四個音訊裝置都指向正確的輸入與輸出。"
        color = "#ef4444" if is_error else "#22c55e"
        self.route_status_label.setText(message)
        self.route_status_label.setStyleSheet(f"color: {color};")

    def _notify_route_changed(self, *_args) -> None:
        if self._suspend_notify:
            return
        self._on_route_changed()

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
        active_hostapi_name = str(hostapi_combo.currentData() or hostapi_name or "")
        self._refill_device_combo(device_combo, devices, active_hostapi_name)
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
        slider: QSlider,
        value_label: QLabel,
    ) -> QWidget:
        hostapi_combo.setMinimumWidth(180)
        hostapi_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        hostapi_combo.setMinimumContentsLength(10)
        hostapi_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hostapi_combo.setToolTip("選擇音訊 API")
        device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        device_combo.setMinimumContentsLength(20)
        device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        device_combo.setToolTip("選擇裝置")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(hostapi_combo, 1)
        row.addWidget(device_combo, 3)
        row.addWidget(AudioRoutingPage._build_gain_control(slider, value_label), 2)
        return container

    @staticmethod
    def _build_gain_control(slider: QSlider, value_label: QLabel) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        slider.setToolTip("真實裝置音量")
        value_label.setMinimumWidth(48)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return container

    @staticmethod
    def _configure_gain_slider(slider: QSlider) -> None:
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setTickInterval(10)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setFixedHeight(30)
        slider.setMinimumWidth(160)
        slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                margin: 0 10px;
                border-radius: 2px;
                background: #d1d5db;
            }
            QSlider::sub-page:horizontal {
                background: #d8b36a;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: #d1d5db;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                height: 18px;
                margin: -8px -9px;
                border-radius: 9px;
                border: 2px solid #4b5563;
                background: #d8b36a;
            }
            QSlider::handle:horizontal:hover {
                background: #e2bf79;
            }
            """
        )

    def _sync_gain_labels(self) -> None:
        self._update_gain_value_label(
            self.meeting_in_gain_value_label,
            self._slider_value_to_gain(self.meeting_in_gain_slider.value()),
        )
        self._update_gain_value_label(
            self.microphone_in_gain_value_label,
            self._slider_value_to_gain(self.microphone_in_gain_slider.value()),
        )
        self._update_gain_value_label(
            self.speaker_out_volume_value_label,
            self._slider_value_to_gain(self.speaker_out_volume_slider.value()),
        )
        self._update_gain_value_label(
            self.meeting_out_volume_value_label,
            self._slider_value_to_gain(self.meeting_out_volume_slider.value()),
        )

    @staticmethod
    def _slider_value_to_gain(value: int) -> float:
        return round(max(0, min(100, int(value))) / 100.0, 2)

    @staticmethod
    def _gain_to_slider_value(value: float) -> int:
        return max(0, min(100, int(round(max(0.0, min(1.0, float(value))) * 100.0))))

    @staticmethod
    def _update_gain_value_label(label: QLabel, gain: float) -> None:
        label.setText(f"{int(round(gain * 100.0))}%")
