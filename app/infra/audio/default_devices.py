from __future__ import annotations

import sounddevice as sd

from app.infra.audio.device_registry import list_indexed_devices, preferred_hostapi_index_for_platform


VIRTUAL_DEVICE_TOKENS = (
    "synctranslate virtual",
    "voicemeeter",
    "vb-audio",
    "virtual cable",
    "cable input",
    "cable output",
)


class SystemDefaultDeviceResolver:
    def __init__(self, *, exclude_virtual_devices: bool = True) -> None:
        self.exclude_virtual_devices = bool(exclude_virtual_devices)

    def default_capture_name(self) -> str:
        return self._system_default_name(kind="capture") or self._first_device_name(kind="capture")

    def default_render_name(self) -> str:
        return self._system_default_name(kind="render") or self._first_device_name(kind="render")

    def _system_default_name(self, *, kind: str) -> str:
        """回傳目前 OS 的預設輸入/輸出裝置名稱。

        這裡不做虛擬裝置過濾，因為使用者期待看到並使用系統當前預設值。
        """
        hostapi_default = self._preferred_hostapi_default_name(kind=kind)
        if hostapi_default:
            return hostapi_default

        fallback = self._global_default_name(kind=kind)
        if fallback and self._is_mapper_name(fallback):
            # On Windows, PortAudio global default may point to MME mapper alias.
            # Prefer endpoint names from the preferred host API when possible.
            preferred = self._preferred_hostapi_default_name(kind=kind)
            if preferred:
                return preferred
        return fallback

    def _preferred_hostapi_default_name(self, *, kind: str) -> str:
        try:
            hostapi_index = preferred_hostapi_index_for_platform()
            if hostapi_index is None:
                return ""
            hostapi = sd.query_hostapis(hostapi_index)
            default_index = hostapi.get("default_input_device") if kind == "capture" else hostapi.get("default_output_device")
            if default_index is None:
                return ""
            return self._device_name_from_index(kind=kind, index=default_index)
        except Exception:
            return ""

    def _global_default_name(self, *, kind: str) -> str:
        try:
            default_pair = sd.default.device
            if not isinstance(default_pair, (tuple, list)) or len(default_pair) < 2:
                return ""
            index = default_pair[0] if kind == "capture" else default_pair[1]
            return self._device_name_from_index(kind=kind, index=index)
        except Exception:
            return ""

    def _device_name_from_index(self, *, kind: str, index: object) -> str:
        try:
            if index is None:
                return ""
            index_int = int(index)
            if index_int < 0:
                return ""
            item = sd.query_devices(index_int)
            name = str(item.get("name") or "").strip()
            if not name:
                return ""
            if kind == "capture" and int(item.get("max_input_channels", 0) or 0) <= 0:
                return ""
            if kind == "render" and int(item.get("max_output_channels", 0) or 0) <= 0:
                return ""
            return name
        except Exception:
            return ""

    @staticmethod
    def _is_mapper_name(name: str) -> bool:
        normalized = name.strip().lower()
        return "microsoft sound mapper" in normalized or "microsoft 音效對應表" in normalized

    def _first_device_name(self, *, kind: str) -> str:
        candidates: list[str] = []
        for _idx, item in list_indexed_devices():
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if self.exclude_virtual_devices and self._is_virtual_device(name):
                continue
            if kind == "capture" and int(item.get("max_input_channels", 0) or 0) <= 0:
                continue
            if kind == "render" and int(item.get("max_output_channels", 0) or 0) <= 0:
                continue
            candidates.append(name)
        return candidates[0] if candidates else ""

    @staticmethod
    def _is_virtual_device(name: str) -> bool:
        normalized = name.strip().lower()
        return any(token in normalized for token in VIRTUAL_DEVICE_TOKENS)


__all__ = ["SystemDefaultDeviceResolver", "VIRTUAL_DEVICE_TOKENS"]
