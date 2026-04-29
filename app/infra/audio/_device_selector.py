"""Output-device selection helpers extracted from AudioPlayback.

All functions are stateless and can be called without an AudioPlayback instance.
"""
from __future__ import annotations

import sounddevice as sd

try:
    import soundcard as sc
except Exception:
    sc = None

from app.infra.audio.device_registry import (
    canonical_device_name,
    device_tokens,
    list_indexed_devices,
    normalize_device_text,
    parse_device_selector,
    preferred_hostapi_index_for_platform,
)


def should_avoid_soundcard_backend(output_device_name: str) -> bool:
    """Return True when the soundcard backend should be skipped for *output_device_name*.

    Virtual audio routers (VoiceMeeter, VB-Audio, generic virtual cables)
    work more reliably through sounddevice/WASAPI than through soundcard.
    """
    normalized = normalize_device_text(canonical_device_name(output_device_name))
    return any(token in normalized for token in ("voicemeeter", "vb audio", "virtual"))


def find_soundcard_speaker(output_device_name: str):
    """Return the best-matching soundcard speaker object, or ``None``.

    Speakers are ranked by name similarity (exact match > normalised match >
    substring > token overlap).  The highest-scoring speaker is returned.
    """
    if sc is None:
        return None

    target_name = canonical_device_name(output_device_name)
    normalized_target = normalize_device_text(target_name)
    target_tokens = device_tokens(target_name)
    ranked: list[tuple[int, int, str, int, object]] = []
    for speaker_index, speaker in enumerate(sc.all_speakers()):
        name = str(getattr(speaker, "name", "") or "")
        normalized_name = normalize_device_text(name)
        name_tokens = device_tokens(name)
        score = 0
        if name == target_name:
            score = 500
        elif normalized_name == normalized_target:
            score = 450
        elif normalized_target and normalized_target in normalized_name:
            score = 350
        elif target_tokens and target_tokens.issubset(name_tokens):
            score = 300 + len(target_tokens)
        elif target_tokens:
            overlap = len(target_tokens & name_tokens)
            if overlap >= max(2, len(target_tokens) - 1):
                score = 200 + overlap
        if score > 0:
            extra_token_penalty = max(0, len(name_tokens) - len(target_tokens))
            ranked.append((-score, extra_token_penalty, normalized_name, speaker_index, speaker))

    ranked.sort()
    return ranked[0][4] if ranked else None


def find_output_devices(device_name: str) -> list[tuple[int, dict[str, object]]]:
    """Return a ranked list of ``(device_index, device_info)`` for *device_name*.

    Devices are ranked by:
    1. Host API preference (explicit selector > platform default > others)
    2. Name similarity score (exact > normalised > substring > token overlap)
    3. Fewest extra tokens (most specific match wins)

    Virtual devices (VoiceMeeter, VB-Audio, cable) are handled specially:
    stale DirectSound entries are tolerated and WASAPI/WDM-KS is preferred.
    """
    hostapi_name, requested_name = parse_device_selector(device_name)
    devices = list_indexed_devices()
    preferred_hostapi = preferred_hostapi_index_for_platform()
    normalized_target = normalize_device_text(requested_name)
    target_tokens = device_tokens(requested_name)
    target_looks_virtual = any(
        token in normalized_target for token in ("voicemeeter", "vb audio", "virtual", "cable")
    )
    ranked: list[tuple[int, int, int, int, dict[str, object]]] = []

    for idx, item in devices:
        if int(item["max_output_channels"]) <= 0:
            continue
        name = str(item["name"])
        normalized_name = normalize_device_text(name)
        name_tokens = device_tokens(name)

        score = 0
        if name == requested_name:
            score = 500
        elif normalized_name == normalized_target:
            score = 450
        elif normalized_target and normalized_target in normalized_name:
            score = 350
        elif target_tokens and target_tokens.issubset(name_tokens):
            score = 300 + len(target_tokens)
        elif target_tokens:
            overlap = len(target_tokens & name_tokens)
            if overlap >= max(2, len(target_tokens) - 1):
                score = 200 + overlap

        if score <= 0:
            continue

        hostapi = int(item.get("hostapi", -1))
        if hostapi_name:
            resolved_hostapi_name = str(sd.query_hostapis()[hostapi].get("name", ""))
            hostapi_matches = resolved_hostapi_name == hostapi_name
            # For virtual devices, tolerate stale DirectSound selection and prefer WASAPI/KS.
            if not hostapi_matches:
                if target_looks_virtual:
                    lowered = resolved_hostapi_name.strip().lower()
                    if lowered not in ("windows wasapi", "windows wdm-ks"):
                        continue
                else:
                    continue

            if target_looks_virtual:
                lowered = resolved_hostapi_name.strip().lower()
                if lowered == "windows wasapi":
                    hostapi_rank = -2
                elif lowered == "windows wdm-ks":
                    hostapi_rank = -1
                elif hostapi_matches:
                    hostapi_rank = 0
                else:
                    hostapi_rank = 1
            else:
                hostapi_rank = 0 if hostapi_matches else 1
        else:
            hostapi_rank = 0 if hostapi == preferred_hostapi else 1
        extra_token_penalty = max(0, len(name_tokens) - len(target_tokens))
        ranked.append((hostapi_rank, -score, extra_token_penalty, idx, item))

    ranked.sort()
    return [(idx, item) for _, _, _, idx, item in ranked]


__all__ = [
    "find_output_devices",
    "find_soundcard_speaker",
    "should_avoid_soundcard_backend",
]
