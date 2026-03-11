from __future__ import annotations

from app.audio_device_selection import hostapi_label, hostapi_name_by_index, list_indexed_devices
from app.schemas import DeviceInfo


class DeviceManager:
    def list_all(self) -> list[DeviceInfo]:
        indexed_devices = list_indexed_devices()
        result: list[DeviceInfo] = []
        for index, item in indexed_devices:
            hostapi_index = int(item.get("hostapi", -1))
            hostapi_name = hostapi_name_by_index(hostapi_index)
            result.append(
                DeviceInfo(
                    index=index,
                    name=str(item["name"]),
                    hostapi_index=hostapi_index,
                    hostapi_name=hostapi_name,
                    hostapi_label=hostapi_label(hostapi_name),
                    max_input_channels=int(item["max_input_channels"]),
                    max_output_channels=int(item["max_output_channels"]),
                    default_samplerate=float(item["default_samplerate"]),
                )
            )
        return result

    def list_input_devices(self) -> list[DeviceInfo]:
        return [d for d in self.list_all() if d.max_input_channels > 0]

    def list_output_devices(self) -> list[DeviceInfo]:
        return [d for d in self.list_all() if d.max_output_channels > 0]

    def find_voicemeeter_devices(self) -> list[DeviceInfo]:
        return [d for d in self.list_all() if "voicemeeter" in d.name.lower()]
