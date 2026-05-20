from __future__ import annotations

from app.infra.audio.device_registry import list_indexed_devices


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
        return self._first_device_name(kind="capture")

    def default_render_name(self) -> str:
        return self._first_device_name(kind="render")

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
