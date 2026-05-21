from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.application.call_translation_policy import SYNC_VIRTUAL_AUDIO
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
        self._latest_device_summary: dict[str, str] = {}
        self._current_audio_config = AudioRouteConfig()
        self._suspend_notify = False

        self.routing_mode_combo = QComboBox()
        self.routing_mode_combo.addItem("SyncTranslate 虛擬音訊", SYNC_VIRTUAL_AUDIO)
        self.routing_mode_combo.setEnabled(False)
        self.routing_mode_combo.setToolTip("目前僅支援 SyncTranslate 虛擬音訊模式")

        self.virtual_device_summary_label = QLabel()
        self.virtual_device_summary_label.setWordWrap(True)
        self.virtual_device_summary_label.setStyleSheet("color: #b7bdc6;")

        form = QFormLayout()
        form.addRow("音訊模式", self.routing_mode_combo)
        form.addRow("目前偵測到的裝置", self.virtual_device_summary_label)

        info_label = QLabel("SyncTranslate 不提供音量控制；請直接使用 Windows 系統音量與通訊軟體音量。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #b7bdc6; font-size: 10pt;")

        self.route_status_label = QLabel("請確認已安裝 SyncTranslate 虛擬音訊驅動，並在通訊軟體選擇虛擬喇叭/麥克風。")
        self.route_status_label.setWordWrap(True)
        self.route_status_label.setStyleSheet("color: #b7bdc6;")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(info_label)
        layout.addWidget(self.route_status_label)
        layout.addStretch(1)

        self._sync_virtual_summary()

    def set_devices(self, input_devices: list[DeviceInfo], output_devices: list[DeviceInfo]) -> None:
        self._input_devices = list(input_devices)
        self._output_devices = list(output_devices)

    def apply_config(self, config: AppConfig) -> None:
        self._current_audio_config = deepcopy(config.audio)
        self._refresh_device_summary_label()

    def selected_audio_routes(self) -> AudioRouteConfig:
        current = deepcopy(self._current_audio_config)
        current.routing_mode = SYNC_VIRTUAL_AUDIO
        # 通話翻譯設定現在由 live_caption_page 決定，不再在此處手動控制
        return AudioRouteConfig(
            meeting_in=current.meeting_in,
            microphone_in=current.microphone_in,
            speaker_out=current.speaker_out,
            meeting_out=current.meeting_out,
            routing_mode=current.routing_mode,
            system_devices=current.system_devices,
            virtual_audio=current.virtual_audio,
            call_translation=current.call_translation,
        )

    def set_validation_message(self, text: str, *, is_error: bool) -> None:
        message = text.strip() or "請確認已安裝 SyncTranslate 虛擬音訊驅動，並在通訊軟體選擇虛擬喇叭/麥克風。"
        color = "#ef4444" if is_error else "#22c55e"
        self.route_status_label.setText(message)
        self.route_status_label.setStyleSheet(f"color: {color};")

    def _notify_route_changed(self, *_args) -> None:
        if self._suspend_notify:
            return
        self._on_route_changed()

    def update_device_summary(self, summary: dict[str, str]) -> None:
        """更新由系統自動偵測的裝置摘要。
        
        Args:
            summary: 包含 'meeting_in', 'microphone_in', 'speaker_out', 'meeting_out' 的字典
        """
        self._latest_device_summary = dict(summary)
        self._refresh_device_summary_label()

    def _sync_virtual_summary(self) -> None:
        self._refresh_device_summary_label()

    def _refresh_device_summary_label(self) -> None:
        microphone_in = str(self._latest_device_summary.get("microphone_in", "") or "(未偵測)")
        speaker_out = str(self._latest_device_summary.get("speaker_out", "") or "(未偵測)")
        self.virtual_device_summary_label.setText(
            f"本地輸入：{microphone_in}；本地輸出：{speaker_out}"
        )
