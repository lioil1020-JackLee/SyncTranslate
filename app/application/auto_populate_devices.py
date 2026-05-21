from __future__ import annotations

from app.infra.audio.default_devices import SystemDefaultDeviceResolver
from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.config.schema import AudioRouteConfig


class AutoPopulateDevicesService:
    """自動填充四個音訊裝置的服務：
    - meeting_in: 虛擬喇叭（遠端音訊輸入）
    - microphone_in: 系統預設通訊麥克風
    - speaker_out: 系統預設通訊揚聲器
    - meeting_out: 虛擬麥克風（本地翻譯輸出）
    """

    def __init__(self, *, exclude_virtual_devices: bool = True) -> None:
        self.system_resolver = SystemDefaultDeviceResolver(exclude_virtual_devices=exclude_virtual_devices)

    def populate(self, config: AudioRouteConfig) -> AudioRouteConfig:
        """根據系統偵測自動填充設備名稱。
        
        Args:
            config: 現有配置
            
        Returns:
            更新後的配置副本，虛擬裝置名稱永遠使用系統當前偵測值
        """
        import logging
        logger = logging.getLogger(__name__)
        
        virtual_status = detect_virtual_audio_install()
        default_capture = self.system_resolver.default_capture_name()
        default_render = self.system_resolver.default_render_name()

        # Always use detected virtual device names to handle driver/system updates.
        # Even if config has old names, these will be refreshed to match current system state.
        config.meeting_in = virtual_status.speaker_name or config.meeting_in or ""
        config.microphone_in = default_capture or config.microphone_in or ""
        config.speaker_out = default_render or config.speaker_out or ""
        config.meeting_out = virtual_status.microphone_name or config.meeting_out or ""
        
        logger.info(
            f"Auto-populated audio routes: "
            f"meeting_in={config.meeting_in!r}, "
            f"meeting_out={config.meeting_out!r}, "
            f"microphone_in={config.microphone_in!r}, "
            f"speaker_out={config.speaker_out!r}"
        )

        return config

    def get_device_summary(self) -> dict[str, str]:
        """取得當前偵測到的裝置摘要。
        
        Returns:
            包含 meeting_in, microphone_in, speaker_out, meeting_out 的字典
        """
        virtual_status = detect_virtual_audio_install()
        default_capture = self.system_resolver.default_capture_name()
        default_render = self.system_resolver.default_render_name()

        return {
            "meeting_in": virtual_status.speaker_name or "(未偵測)",
            "microphone_in": default_capture or "(未偵測)",
            "speaker_out": default_render or "(未偵測)",
            "meeting_out": virtual_status.microphone_name or "(未偵測)",
        }


__all__ = ["AutoPopulateDevicesService"]
