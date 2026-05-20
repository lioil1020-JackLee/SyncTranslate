from __future__ import annotations

from dataclasses import dataclass

from app.infra.audio.device_registry import hostapi_name_by_index, list_indexed_devices


SYNC_DEVICE_TOKEN = "synctranslate"
SYNC_SPEAKER_TOKEN = "virtual speaker"
SYNC_MICROPHONE_TOKEN = "virtual microphone"


@dataclass(frozen=True, slots=True)
class VirtualAudioEndpoint:
    index: int
    name: str
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_capture(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_render(self) -> bool:
        return self.max_output_channels > 0


@dataclass(frozen=True, slots=True)
class VirtualAudioInstallStatus:
    installed: bool
    speaker_available: bool
    microphone_available: bool
    speaker_index: int
    microphone_index: int
    speaker_name: str
    microphone_name: str
    render_endpoints: tuple[VirtualAudioEndpoint, ...]
    capture_endpoints: tuple[VirtualAudioEndpoint, ...]


def detect_virtual_audio_install() -> VirtualAudioInstallStatus:
    endpoints = _sync_endpoints()
    render_endpoints = tuple(endpoint for endpoint in endpoints if endpoint.is_render)
    capture_endpoints = tuple(endpoint for endpoint in endpoints if endpoint.is_capture)
    speaker = _best_render_endpoint(render_endpoints)
    microphone = _best_capture_endpoint(capture_endpoints)
    return VirtualAudioInstallStatus(
        installed=bool(render_endpoints and capture_endpoints),
        speaker_available=speaker is not None,
        microphone_available=microphone is not None,
        speaker_index=speaker.index if speaker else -1,
        microphone_index=microphone.index if microphone else -1,
        speaker_name=speaker.name if speaker else "",
        microphone_name=microphone.name if microphone else "",
        render_endpoints=render_endpoints,
        capture_endpoints=capture_endpoints,
    )


def _sync_endpoints() -> tuple[VirtualAudioEndpoint, ...]:
    result: list[VirtualAudioEndpoint] = []
    seen: set[tuple[str, str, int, int, int]] = set()
    try:
        indexed_devices = list_indexed_devices()
    except Exception:
        return ()
    for index, item in indexed_devices:
        name = str(item.get("name") or "").strip()
        if SYNC_DEVICE_TOKEN not in name.lower():
            continue
        hostapi_index = int(item.get("hostapi", -1) or -1)
        hostapi_name = hostapi_name_by_index(hostapi_index)
        max_input_channels = int(item.get("max_input_channels", 0) or 0)
        max_output_channels = int(item.get("max_output_channels", 0) or 0)
        default_samplerate = float(item.get("default_samplerate", 0.0) or 0.0)
        key = (
            hostapi_name.lower(),
            name.lower(),
            max_input_channels,
            max_output_channels,
            int(default_samplerate),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            VirtualAudioEndpoint(
                index=index,
                name=name,
                hostapi_name=hostapi_name,
                max_input_channels=max_input_channels,
                max_output_channels=max_output_channels,
                default_samplerate=default_samplerate,
            )
        )
    return tuple(sorted(result, key=_endpoint_sort_key))


def _best_render_endpoint(endpoints: tuple[VirtualAudioEndpoint, ...]) -> VirtualAudioEndpoint | None:
    if not endpoints:
        return None
    return min(endpoints, key=lambda endpoint: _named_endpoint_score(endpoint, SYNC_SPEAKER_TOKEN))


def _best_capture_endpoint(endpoints: tuple[VirtualAudioEndpoint, ...]) -> VirtualAudioEndpoint | None:
    if not endpoints:
        return None
    return min(endpoints, key=lambda endpoint: _named_endpoint_score(endpoint, SYNC_MICROPHONE_TOKEN))


def _named_endpoint_score(endpoint: VirtualAudioEndpoint, preferred_token: str) -> tuple[int, int, int, str]:
    normalized = endpoint.name.lower()
    preferred_name_score = 0 if preferred_token in normalized else 1
    semantic_score = _semantic_endpoint_score(normalized, preferred_token)
    hostapi_score = _hostapi_score(endpoint.hostapi_name)
    sample_rate_score = 0 if int(endpoint.default_samplerate) == 48000 else 1
    return (hostapi_score, preferred_name_score, semantic_score, sample_rate_score, normalized)


def _semantic_endpoint_score(normalized_name: str, preferred_token: str) -> int:
    if preferred_token == SYNC_SPEAKER_TOKEN:
        if "喇叭" in normalized_name or "speaker" in normalized_name:
            return 0
        if "headphone" in normalized_name or "耳機" in normalized_name or "output" in normalized_name:
            return 1
        if "spdif" in normalized_name or "sinkdescription" in normalized_name:
            return 4
        return 2
    if preferred_token == SYNC_MICROPHONE_TOKEN:
        if "microphone" in normalized_name or "麥克風" in normalized_name:
            return 0
        if "input" in normalized_name:
            return 1
        return 2
    return 2


def _endpoint_sort_key(endpoint: VirtualAudioEndpoint) -> tuple[int, str, str, int]:
    return (_hostapi_score(endpoint.hostapi_name), endpoint.name.lower(), endpoint.hostapi_name.lower(), endpoint.index)


def _hostapi_score(hostapi_name: str) -> int:
    normalized = (hostapi_name or "").strip().lower()
    if normalized == "windows wasapi":
        return 0
    if normalized == "windows wdm-ks":
        return 1
    if normalized == "mme":
        return 2
    if normalized == "windows directsound":
        return 3
    return 9


__all__ = [
    "VirtualAudioEndpoint",
    "VirtualAudioInstallStatus",
    "detect_virtual_audio_install",
]
