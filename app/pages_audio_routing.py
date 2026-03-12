from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from app.audio_device_selection import canonical_device_name, encode_device_selector, hostapi_sort_key, parse_device_selector
from app.schemas import AppConfig, AudioRouteConfig, DeviceInfo
from app.windows_volume import get_input_volume, get_output_volume


HARDCODED_MEETING_IN = "Windows WASAPI::CABLE Output (VB-Audio Virtual Cable)"
HARDCODED_MEETING_OUT = "Windows WASAPI::CABLE Input (VB-Audio Virtual Cable)"


class VolumeSlider(QSlider):
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(100)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )

        text = f"{self.value()}%"
        painter = QPainter(self)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        x = handle_rect.center().x() - (text_width // 2)
        x = max(0, min(self.width() - text_width, x))
        y = max(metrics.ascent() + 2, handle_rect.top() - 4)
        painter.drawText(x, y, text)
        painter.end()


class AudioRoutingPage(QWidget):
    def __init__(
        self,
        on_route_changed,
    ) -> None:
        super().__init__()
        self._on_route_changed = on_route_changed
        self._input_devices: list[DeviceInfo] = []
        self._output_devices: list[DeviceInfo] = []
        self._system_volume_initialized = False

        self.meeting_in_fixed_label = QLabel(canonical_device_name(HARDCODED_MEETING_IN))
        self.microphone_in_hostapi_combo = QComboBox()
        self.microphone_in_combo = QComboBox()
        self.microphone_in_volume_slider = self._create_volume_slider()
        self.speaker_out_hostapi_combo = QComboBox()
        self.speaker_out_combo = QComboBox()
        self.speaker_out_volume_slider = self._create_volume_slider()
        self.meeting_out_fixed_label = QLabel(canonical_device_name(HARDCODED_MEETING_OUT))
        self.meeting_in_volume_slider = self._create_volume_slider()
        self.meeting_out_volume_slider = self._create_volume_slider()

        for combo in (
            self.microphone_in_hostapi_combo,
            self.microphone_in_combo,
            self.speaker_out_hostapi_combo,
            self.speaker_out_combo,
        ):
            combo.currentIndexChanged.connect(self._on_route_changed)

        for slider in (
            self.meeting_in_volume_slider,
            self.meeting_out_volume_slider,
            self.microphone_in_volume_slider,
            self.speaker_out_volume_slider,
        ):
            slider.valueChanged.connect(self._on_route_changed)

        self.microphone_in_hostapi_combo.currentIndexChanged.connect(self._refresh_microphone_in_combo)
        self.speaker_out_hostapi_combo.currentIndexChanged.connect(self._refresh_speaker_out_combo)

        form = QFormLayout()
        form.addRow(
            "遠端輸入",
            self._build_fixed_route_row(
                self.meeting_in_fixed_label,
                self.meeting_in_volume_slider,
            ),
        )
        form.addRow(
            "遠端輸出",
            self._build_fixed_route_row(
                self.meeting_out_fixed_label,
                self.meeting_out_volume_slider,
            ),
        )
        form.addRow(
            "本地輸入",
            self._build_route_selector_row(
                self.microphone_in_hostapi_combo,
                self.microphone_in_combo,
                self.microphone_in_volume_slider,
            ),
        )
        form.addRow(
            "本地輸出",
            self._build_route_selector_row(
                self.speaker_out_hostapi_combo,
                self.speaker_out_combo,
                self.speaker_out_volume_slider,
            ),
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)
        current = self.selected_audio_routes()
        self.meeting_in_fixed_label.setText(canonical_device_name(HARDCODED_MEETING_IN))
        self.meeting_out_fixed_label.setText(canonical_device_name(HARDCODED_MEETING_OUT))

        self._fill_hostapi_combo(self.microphone_in_hostapi_combo, self._input_devices)
        self._fill_hostapi_combo(self.speaker_out_hostapi_combo, self._output_devices)
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

    def apply_config(self, config: AppConfig) -> None:
        config.audio.meeting_in = HARDCODED_MEETING_IN
        config.audio.meeting_out = HARDCODED_MEETING_OUT
        self._initialize_system_volume(config)
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
        self.meeting_in_fixed_label.setText(canonical_device_name(HARDCODED_MEETING_IN))
        self.meeting_out_fixed_label.setText(canonical_device_name(HARDCODED_MEETING_OUT))
        self._set_slider_ratio(self.meeting_in_volume_slider, config.audio.meeting_in_gain)
        self._set_slider_ratio(self.microphone_in_volume_slider, config.audio.microphone_in_gain)
        self._set_slider_ratio(self.speaker_out_volume_slider, config.audio.speaker_out_volume)
        self._set_slider_ratio(self.meeting_out_volume_slider, config.audio.meeting_out_volume)

    def _initialize_system_volume(self, config: AppConfig) -> None:
        if self._system_volume_initialized:
            return
        meeting_input_volume = None if self._should_skip_system_sync(HARDCODED_MEETING_IN) else get_input_volume(HARDCODED_MEETING_IN)
        microphone_volume = None if self._should_skip_system_sync(config.audio.microphone_in) else get_input_volume(config.audio.microphone_in)
        speaker_volume = None if self._should_skip_system_sync(config.audio.speaker_out) else get_output_volume(config.audio.speaker_out)
        meeting_output_volume = None if self._should_skip_system_sync(HARDCODED_MEETING_OUT) else get_output_volume(HARDCODED_MEETING_OUT)
        if meeting_input_volume is not None:
            config.audio.meeting_in_gain = meeting_input_volume
        if microphone_volume is not None:
            config.audio.microphone_in_gain = microphone_volume
        if speaker_volume is not None:
            config.audio.speaker_out_volume = speaker_volume
        if meeting_output_volume is not None:
            config.audio.meeting_out_volume = meeting_output_volume
        self._system_volume_initialized = True

    def sync_system_volume_sliders(self, levels: dict[str, float | None]) -> bool:
        changed = False
        meeting_input_volume = levels.get("meeting_in")
        microphone_volume = levels.get("microphone_in")
        speaker_volume = levels.get("speaker_out")
        meeting_output_volume = levels.get("meeting_out")
        if meeting_input_volume is not None:
            changed |= self._set_slider_ratio_silently(
                self.meeting_in_volume_slider,
                meeting_input_volume,
            )
        if microphone_volume is not None:
            changed |= self._set_slider_ratio_silently(
                self.microphone_in_volume_slider,
                microphone_volume,
            )
        if speaker_volume is not None:
            changed |= self._set_slider_ratio_silently(
                self.speaker_out_volume_slider,
                speaker_volume,
            )
        if meeting_output_volume is not None:
            changed |= self._set_slider_ratio_silently(
                self.meeting_out_volume_slider,
                meeting_output_volume,
            )
        return changed

    def selected_audio_routes(self) -> AudioRouteConfig:
        return AudioRouteConfig(
            meeting_in=HARDCODED_MEETING_IN,
            microphone_in=self._selected_selector(self.microphone_in_combo),
            speaker_out=self._selected_selector(self.speaker_out_combo),
            meeting_out=HARDCODED_MEETING_OUT,
            meeting_in_gain=self._slider_ratio(self.meeting_in_volume_slider),
            microphone_in_gain=self._slider_ratio(self.microphone_in_volume_slider),
            speaker_out_volume=self._slider_ratio(self.speaker_out_volume_slider),
            meeting_out_volume=self._slider_ratio(self.meeting_out_volume_slider),
        )

    def _refresh_microphone_in_combo(self) -> None:
        self._refill_device_combo(self.microphone_in_combo, self._input_devices, self.microphone_in_hostapi_combo.currentData())

    def _refresh_speaker_out_combo(self) -> None:
        self._refill_device_combo(self.speaker_out_combo, self._output_devices, self.speaker_out_hostapi_combo.currentData())

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
            label = f"[{device.index}] {device.name}"
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
    def _build_route_selector_row(
        hostapi_combo: QComboBox,
        device_combo: QComboBox,
        volume_slider: QSlider,
    ) -> QWidget:
        hostapi_combo.setMinimumWidth(180)
        hostapi_combo.setToolTip("音訊介面")
        device_combo.setToolTip("音訊裝置")
        volume_slider.setToolTip("音量")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(hostapi_combo, 1)
        row.addWidget(device_combo, 2)
        row.addWidget(volume_slider, 1)
        return container

    @staticmethod
    def _build_fixed_route_row(
        device_label: QLabel,
        volume_slider: QSlider,
    ) -> QWidget:
        device_label.setMinimumWidth(180)
        device_label.setToolTip("固定音訊裝置")
        volume_slider.setToolTip("音量")
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(device_label, 3)
        row.addWidget(volume_slider, 1)
        return container

    @staticmethod
    def _create_volume_slider() -> QSlider:
        slider = VolumeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 200)
        slider.setSingleStep(5)
        slider.setPageStep(10)
        slider.setValue(100)
        return slider

    @staticmethod
    def _slider_ratio(slider: QSlider) -> float:
        return float(slider.value()) / 100.0

    @staticmethod
    def _set_slider_ratio(slider: QSlider, value: float) -> None:
        slider.setValue(max(0, min(200, int(round(float(value) * 100.0)))))

    @staticmethod
    def _set_slider_ratio_silently(slider: QSlider, value: float) -> bool:
        target = max(0, min(200, int(round(float(value) * 100.0))))
        if slider.value() == target:
            return False
        slider.blockSignals(True)
        slider.setValue(target)
        slider.blockSignals(False)
        return True

    @staticmethod
    def _should_skip_system_sync(device_selector: str) -> bool:
        name = canonical_device_name(device_selector).lower()
        return any(token in name for token in ("voicemeeter", "vb-audio", "virtual"))
