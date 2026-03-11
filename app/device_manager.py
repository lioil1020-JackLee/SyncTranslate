from __future__ import annotations

import sounddevice as sd

from app.schemas import DeviceInfo


class DeviceManager:
    def list_all(self) -> list[DeviceInfo]:
        devices = sd.query_devices()
        result: list[DeviceInfo] = []
        for index, item in enumerate(devices):
            result.append(
                DeviceInfo(
                    index=index,
                    name=str(item["name"]),
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

