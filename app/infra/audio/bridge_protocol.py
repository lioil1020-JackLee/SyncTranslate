from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np


BRIDGE_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class BridgeResponse:
    ok: bool
    payload: dict[str, Any]
    error: str = ""


def encode_audio_packet(audio: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
    payload = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    return {
        "sample_rate": int(sample_rate),
        "shape": list(payload.shape),
        "dtype": "float32",
        "data_b64": base64.b64encode(payload.tobytes()).decode("ascii"),
    }


def decode_audio_packet(packet: dict[str, Any]) -> tuple[np.ndarray, int]:
    sample_rate = int(packet.get("sample_rate", 0) or 0)
    shape = packet.get("shape") or []
    if not isinstance(shape, list) or not shape:
        raise ValueError("audio packet shape is required")
    raw = base64.b64decode(str(packet.get("data_b64") or ""))
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    audio = audio.reshape(tuple(int(item) for item in shape))
    return audio, sample_rate


def audio_frame_count(audio: np.ndarray) -> int:
    if audio.ndim == 0:
        return int(audio.size)
    return int(audio.shape[0])


def audio_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "BridgeResponse",
    "audio_frame_count",
    "audio_peak",
    "decode_audio_packet",
    "encode_audio_packet",
]
