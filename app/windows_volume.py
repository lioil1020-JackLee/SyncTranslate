from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time

from app.audio_device_selection import canonical_device_name, normalize_device_text

try:
    from pycaw.pycaw import AudioUtilities
except Exception:  # pragma: no cover - optional runtime dependency
    AudioUtilities = None


@dataclass(slots=True)
class _CachedDevice:
    friendly_name: str
    endpoint: object


_CACHE_LOCK = Lock()
_DEVICE_CACHE: list[_CachedDevice] = []
_CACHE_EXPIRES_AT = 0.0
_CACHE_TTL_SEC = 5.0


def get_output_volume(device_selector: str = "") -> float | None:
    return _get_endpoint_volume(device_selector=device_selector, is_input=False)


def get_input_volume(device_selector: str = "") -> float | None:
    return _get_endpoint_volume(device_selector=device_selector, is_input=True)


def _get_endpoint_volume(*, device_selector: str, is_input: bool) -> float | None:
    if AudioUtilities is None:
        return None
    device = _find_matching_device(device_selector=device_selector, is_input=is_input)
    if device is None:
        device = _get_default_device(is_input=is_input)
    if device is None:
        return None
    try:
        return max(0.0, min(1.0, float(device.EndpointVolume.GetMasterVolumeLevelScalar())))
    except Exception:
        return None


def _get_default_device(*, is_input: bool):
    if AudioUtilities is None:
        return None
    try:
        return AudioUtilities.GetMicrophone() if is_input else AudioUtilities.GetSpeakers()
    except Exception:
        return None


def _find_matching_device(*, device_selector: str, is_input: bool):
    if AudioUtilities is None:
        return None
    target = normalize_device_text(canonical_device_name(device_selector))
    if not target:
        return None
    best = None
    best_score = -1
    for device in _list_cached_devices():
        normalized_name = normalize_device_text(device.friendly_name)
        if not normalized_name:
            continue
        score = _match_score(target, normalized_name)
        if score <= 0:
            continue
        if best is None or score > best_score:
            best = device.endpoint
            best_score = score
    return best


def _match_score(target: str, candidate: str) -> int:
    if candidate == target:
        return 500
    if target and target in candidate:
        return 400
    target_tokens = set(target.split())
    candidate_tokens = set(candidate.split())
    if target_tokens and target_tokens.issubset(candidate_tokens):
        return 300 + len(target_tokens)
    overlap = len(target_tokens & candidate_tokens)
    if overlap:
        return 200 + overlap
    return 0


def _list_cached_devices() -> list[_CachedDevice]:
    global _DEVICE_CACHE, _CACHE_EXPIRES_AT
    now = time.monotonic()
    with _CACHE_LOCK:
        if _DEVICE_CACHE and now < _CACHE_EXPIRES_AT:
            return list(_DEVICE_CACHE)
    devices: list[_CachedDevice] = []
    if AudioUtilities is None:
        return devices
    try:
        raw_devices = AudioUtilities.GetAllDevices()
    except Exception:
        return devices
    for device in raw_devices:
        try:
            friendly_name = str(getattr(device, "FriendlyName", "") or "")
            endpoint = device.EndpointVolume
        except Exception:
            continue
        if not friendly_name:
            continue
        devices.append(_CachedDevice(friendly_name=friendly_name, endpoint=endpoint))
    with _CACHE_LOCK:
        _DEVICE_CACHE = devices
        _CACHE_EXPIRES_AT = now + _CACHE_TTL_SEC
    return list(devices)
