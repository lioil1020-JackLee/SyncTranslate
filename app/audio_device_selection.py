from __future__ import annotations

import platform
from collections.abc import Sequence

import sounddevice as sd

IndexedDevice = tuple[int, dict[str, object]]


def list_indexed_devices() -> list[IndexedDevice]:
    return [(idx, item) for idx, item in enumerate(sd.query_devices())]


def select_best_hostapi_devices(indexed_devices: Sequence[IndexedDevice]) -> list[IndexedDevice]:
    preferred_hostapi_index = preferred_hostapi_index_for_platform()
    if preferred_hostapi_index is None:
        return []
    return [pair for pair in indexed_devices if int(pair[1].get("hostapi", -1)) == preferred_hostapi_index]


def pick_best_device(indexed_devices: Sequence[IndexedDevice]) -> IndexedDevice | None:
    preferred_hostapi_index = preferred_hostapi_index_for_platform()
    if preferred_hostapi_index is None:
        return None
    candidates = [pair for pair in indexed_devices if int(pair[1].get("hostapi", -1)) == preferred_hostapi_index]
    return min(candidates, key=lambda pair: pair[0]) if candidates else None


def preferred_hostapi_index_for_platform() -> int | None:
    preferred_name = _preferred_hostapi_name()
    if not preferred_name:
        return None
    for index, item in enumerate(sd.query_hostapis()):
        if str(item.get("name", "")) == preferred_name:
            return index
    return None


def _preferred_hostapi_name() -> str | None:
    system = platform.system().lower()
    if system == "windows":
        return "Windows WASAPI"
    if system == "darwin":
        return "Core Audio"
    return "PulseAudio"
