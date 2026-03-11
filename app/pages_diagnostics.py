from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio_capture import CaptureStats
from app.schemas import DeviceInfo


class DiagnosticsPage(QWidget):
    def __init__(
        self,
        on_start_remote_capture: Callable[[str], None],
        on_stop_remote_capture: Callable[[], None],
        on_start_local_capture: Callable[[str], None],
        on_stop_local_capture: Callable[[], None],
        on_set_local_mute: Callable[[bool], None],
        on_rebind_local_capture: Callable[[], None],
        on_test_meeting_tts: Callable[[], None],
        on_export_diagnostics: Callable[[], None],
        remote_stats_provider: Callable[[], CaptureStats],
        local_stats_provider: Callable[[], CaptureStats],
    ) -> None:
        super().__init__()
        self._on_start_remote_capture = on_start_remote_capture
        self._on_stop_remote_capture = on_stop_remote_capture
        self._on_start_local_capture = on_start_local_capture
        self._on_stop_local_capture = on_stop_local_capture
        self._on_set_local_mute = on_set_local_mute
        self._on_rebind_local_capture = on_rebind_local_capture
        self._on_test_meeting_tts = on_test_meeting_tts
        self._on_export_diagnostics = on_export_diagnostics
        self._remote_stats_provider = remote_stats_provider
        self._local_stats_provider = local_stats_provider

        self.remote_in_combo = QComboBox()
        self.local_mic_combo = QComboBox()

        self.remote_sample_rate_label = QLabel("0")
        self.remote_frame_count_label = QLabel("0")
        self.remote_level_bar = QProgressBar()
        self.remote_level_bar.setRange(0, 100)
        self.remote_status_label = QLabel("停止")

        self.local_sample_rate_label = QLabel("0")
        self.local_frame_count_label = QLabel("0")
        self.local_level_bar = QProgressBar()
        self.local_level_bar.setRange(0, 100)
        self.local_status_label = QLabel("停止")
        self.local_mute_checkbox = QCheckBox("靜音 local_mic")
        self.local_mute_checkbox.toggled.connect(self._on_set_local_mute)

        remote_box = QGroupBox("remote_in 擷取")
        remote_form = QFormLayout()
        remote_form.addRow("裝置", self.remote_in_combo)
        remote_form.addRow("sample rate", self.remote_sample_rate_label)
        remote_form.addRow("frame 計數", self.remote_frame_count_label)
        remote_form.addRow("即時音量", self.remote_level_bar)
        remote_form.addRow("狀態", self.remote_status_label)
        remote_buttons = QHBoxLayout()
        remote_start_btn = QPushButton("開始 remote 擷取")
        remote_stop_btn = QPushButton("停止 remote 擷取")
        remote_start_btn.clicked.connect(self._start_remote_capture_clicked)
        remote_stop_btn.clicked.connect(self._on_stop_remote_capture)
        remote_buttons.addWidget(remote_start_btn)
        remote_buttons.addWidget(remote_stop_btn)
        remote_layout = QVBoxLayout()
        remote_layout.addLayout(remote_form)
        remote_layout.addLayout(remote_buttons)
        remote_box.setLayout(remote_layout)

        local_box = QGroupBox("local_mic 擷取")
        local_form = QFormLayout()
        local_form.addRow("裝置", self.local_mic_combo)
        local_form.addRow("sample rate", self.local_sample_rate_label)
        local_form.addRow("frame 計數", self.local_frame_count_label)
        local_form.addRow("即時音量", self.local_level_bar)
        local_form.addRow("狀態", self.local_status_label)
        local_form.addRow(self.local_mute_checkbox)
        local_buttons = QHBoxLayout()
        local_start_btn = QPushButton("開始 local 擷取")
        local_stop_btn = QPushButton("停止 local 擷取")
        local_rebind_btn = QPushButton("重新綁定 local")
        local_test_btn = QPushButton("測試英文送出")
        export_btn = QPushButton("匯出診斷資訊")
        local_start_btn.clicked.connect(self._start_local_capture_clicked)
        local_stop_btn.clicked.connect(self._on_stop_local_capture)
        local_rebind_btn.clicked.connect(self._on_rebind_local_capture)
        local_test_btn.clicked.connect(self._on_test_meeting_tts)
        export_btn.clicked.connect(self._on_export_diagnostics)
        local_buttons.addWidget(local_start_btn)
        local_buttons.addWidget(local_stop_btn)
        local_buttons.addWidget(local_rebind_btn)
        local_buttons.addWidget(local_test_btn)
        local_buttons.addWidget(export_btn)
        local_layout = QVBoxLayout()
        local_layout.addLayout(local_form)
        local_layout.addLayout(local_buttons)
        local_box.setLayout(local_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(remote_box)
        layout.addWidget(local_box)
        layout.addStretch(1)

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start()

    def set_input_devices(self, input_devices: list[DeviceInfo]) -> None:
        self._fill_combo(self.remote_in_combo, input_devices)
        self._fill_combo(self.local_mic_combo, input_devices)

    def select_remote_in(self, name: str) -> None:
        index = self.remote_in_combo.findData(name)
        self.remote_in_combo.setCurrentIndex(index if index >= 0 else 0)

    def select_local_mic(self, name: str) -> None:
        index = self.local_mic_combo.findData(name)
        self.local_mic_combo.setCurrentIndex(index if index >= 0 else 0)

    def selected_remote_device_name(self) -> str:
        value = self.remote_in_combo.currentData()
        return str(value) if value else ""

    def selected_local_device_name(self) -> str:
        value = self.local_mic_combo.currentData()
        return str(value) if value else ""

    def refresh_stats(self) -> None:
        remote_stats = self._remote_stats_provider()
        local_stats = self._local_stats_provider()
        self._apply_stats(
            remote_stats,
            self.remote_sample_rate_label,
            self.remote_frame_count_label,
            self.remote_level_bar,
            self.remote_status_label,
        )
        self._apply_stats(
            local_stats,
            self.local_sample_rate_label,
            self.local_frame_count_label,
            self.local_level_bar,
            self.local_status_label,
        )
        if self.local_mute_checkbox.isChecked() != local_stats.muted:
            self.local_mute_checkbox.blockSignals(True)
            self.local_mute_checkbox.setChecked(local_stats.muted)
            self.local_mute_checkbox.blockSignals(False)

    def _start_remote_capture_clicked(self) -> None:
        self._on_start_remote_capture(self.selected_remote_device_name())

    def _start_local_capture_clicked(self) -> None:
        self._on_start_local_capture(self.selected_local_device_name())

    @staticmethod
    def _fill_combo(combo: QComboBox, input_devices: list[DeviceInfo]) -> None:
        current_name = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", "")
        for device in input_devices:
            combo.addItem(f"[{device.index}] {device.name}", device.name)
        if current_name:
            index = combo.findData(current_name)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _apply_stats(
        stats: CaptureStats,
        sample_rate_label: QLabel,
        frame_count_label: QLabel,
        level_bar: QProgressBar,
        status_label: QLabel,
    ) -> None:
        sample_rate_label.setText(f"{stats.sample_rate:.0f}")
        frame_count_label.setText(str(stats.frame_count))
        level_bar.setValue(max(0, min(100, int(stats.level * 1000))))
        if stats.last_error:
            status_label.setText(f"Error: {stats.last_error}")
        else:
            mute_suffix = " (靜音)" if stats.muted else ""
            status_label.setText(("執行中" if stats.running else "停止") + mute_suffix)
