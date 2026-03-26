from __future__ import annotations

import platform
import re
from collections.abc import Sequence

import sounddevice as sd

from app.infra.config.schema import DeviceInfo

IndexedDevice = tuple[int, dict[str, object]]


def list_indexed_devices() -> list[IndexedDevice]:
    return [(idx, item) for idx, item in enumerate(sd.query_devices())]


def hostapi_name_by_index(index: int) -> str:
    hostapis = sd.query_hostapis()
    if 0 <= index < len(hostapis):
        return str(hostapis[index].get("name", ""))
    return ""


def hostapi_label(hostapi_name: str) -> str:
    normalized = (hostapi_name or "").strip().lower()
    mapping = {
        "windows wasapi": "WDM (WASAPI)",
        "windows wdm-ks": "KS (Kernel Streaming)",
        "mme": "MME (Multimedia)",
        "windows directsound": "DirectSound",
        "asio": "ASIO",
    }
    return mapping.get(normalized, hostapi_name or "Unknown")


def hostapi_sort_key(hostapi_name: str) -> tuple[int, str]:
    normalized = (hostapi_name or "").strip().lower()
    priority = {
        "windows wasapi": 0,
        "windows wdm-ks": 1,
        "mme": 2,
        "windows directsound": 3,
        "asio": 4,
    }
    return (priority.get(normalized, 99), normalized)


def encode_device_selector(*, hostapi_name: str, device_name: str) -> str:
    if not hostapi_name.strip():
        return device_name
    return f"{hostapi_name}::{device_name}"


def parse_device_selector(selector: str) -> tuple[str, str]:
    value = (selector or "").strip()
    if "::" in value:
        hostapi_name, device_name = value.split("::", 1)
        return hostapi_name.strip(), device_name.strip()
    return "", value


def canonical_device_name(selector: str) -> str:
    _, device_name = parse_device_selector(selector)
    return device_name


def normalize_device_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", lowered)
    return " ".join(cleaned.split())


def device_tokens(value: str) -> set[str]:
    normalized = normalize_device_text(value)
    return {token for token in normalized.split(" ") if token}


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


__all__ = [
    "IndexedDevice",
    "list_indexed_devices",
    "hostapi_name_by_index",
    "hostapi_label",
    "hostapi_sort_key",
    "encode_device_selector",
    "parse_device_selector",
    "canonical_device_name",
    "normalize_device_text",
    "device_tokens",
    "select_best_hostapi_devices",
    "pick_best_device",
    "preferred_hostapi_index_for_platform",
    "DeviceManager",
    "DeviceInfo",
]
