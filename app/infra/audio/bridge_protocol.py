from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.infra.asr.resampling import resample_audio


BRIDGE_PROTOCOL_VERSION = 1
VIRTUAL_AUDIO_PROTOCOL_V2 = 2
VIRTUAL_AUDIO_V2_SAMPLE_RATE = 48000
VIRTUAL_AUDIO_V2_CHANNELS = 2
VIRTUAL_AUDIO_V2_BIT_DEPTH = 16
VIRTUAL_AUDIO_V2_DTYPE = "int16"
VIRTUAL_AUDIO_V2_LAYOUT = "interleaved_stereo"


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


def encode_pcm16_stereo_packet(audio: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
    stereo = _to_stereo_float32_48k(audio, sample_rate=int(sample_rate))
    pcm = np.clip(stereo, -1.0, 1.0)
    payload = np.ascontiguousarray((pcm * 32767.0).astype("<i2", copy=False))
    frames = int(payload.shape[0])
    return {
        "protocol_version": VIRTUAL_AUDIO_PROTOCOL_V2,
        "sample_rate": VIRTUAL_AUDIO_V2_SAMPLE_RATE,
        "channels": VIRTUAL_AUDIO_V2_CHANNELS,
        "bit_depth": VIRTUAL_AUDIO_V2_BIT_DEPTH,
        "dtype": VIRTUAL_AUDIO_V2_DTYPE,
        "layout": VIRTUAL_AUDIO_V2_LAYOUT,
        "frames": frames,
        "data_b64": base64.b64encode(payload.tobytes()).decode("ascii"),
    }


def decode_pcm16_stereo_packet(packet: dict[str, Any]) -> tuple[np.ndarray, int]:
    _validate_pcm16_stereo_packet_header(packet)
    frames = int(packet.get("frames", 0) or 0)
    raw = base64.b64decode(str(packet.get("data_b64") or ""), validate=True)
    expected_bytes = frames * VIRTUAL_AUDIO_V2_CHANNELS * np.dtype("<i2").itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"pcm16 stereo packet data length mismatch: expected {expected_bytes} bytes, got {len(raw)}"
        )
    pcm = np.frombuffer(raw, dtype="<i2").copy()
    try:
        pcm = pcm.reshape((frames, VIRTUAL_AUDIO_V2_CHANNELS))
    except ValueError as exc:
        raise ValueError("pcm16 stereo packet frames/data length are inconsistent") from exc
    audio = (pcm.astype(np.float32) / 32767.0).astype(np.float32, copy=False)
    return audio, VIRTUAL_AUDIO_V2_SAMPLE_RATE


def _validate_pcm16_stereo_packet_header(packet: dict[str, Any]) -> None:
    protocol_version = int(packet.get("protocol_version", 0) or 0)
    if protocol_version != VIRTUAL_AUDIO_PROTOCOL_V2:
        raise ValueError(f"unsupported virtual audio protocol_version: {protocol_version}")
    sample_rate = int(packet.get("sample_rate", 0) or 0)
    if sample_rate != VIRTUAL_AUDIO_V2_SAMPLE_RATE:
        raise ValueError(f"unsupported virtual audio sample_rate: {sample_rate}")
    channels = int(packet.get("channels", 0) or 0)
    if channels != VIRTUAL_AUDIO_V2_CHANNELS:
        raise ValueError(f"unsupported virtual audio channels: {channels}")
    bit_depth = int(packet.get("bit_depth", 0) or 0)
    if bit_depth != VIRTUAL_AUDIO_V2_BIT_DEPTH:
        raise ValueError(f"unsupported virtual audio bit_depth: {bit_depth}")
    dtype = str(packet.get("dtype") or "")
    if dtype != VIRTUAL_AUDIO_V2_DTYPE:
        raise ValueError(f"unsupported virtual audio dtype: {dtype}")
    layout = str(packet.get("layout") or "")
    if layout != VIRTUAL_AUDIO_V2_LAYOUT:
        raise ValueError(f"unsupported virtual audio layout: {layout}")
    frames = int(packet.get("frames", -1) or 0)
    if frames < 0:
        raise ValueError("virtual audio packet frames must be non-negative")
    if "data_b64" not in packet:
        raise ValueError("virtual audio packet data_b64 is required")


def _to_stereo_float32_48k(audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
    payload = np.asarray(audio, dtype=np.float32)
    if payload.size == 0:
        return np.zeros((0, VIRTUAL_AUDIO_V2_CHANNELS), dtype=np.float32)
    if payload.ndim == 1:
        payload = payload.reshape((-1, 1))
    elif payload.ndim != 2:
        raise ValueError("audio must be a 1D mono or 2D frame/channel array")
    if payload.shape[1] == 1:
        stereo = np.repeat(payload, VIRTUAL_AUDIO_V2_CHANNELS, axis=1)
    elif payload.shape[1] >= VIRTUAL_AUDIO_V2_CHANNELS:
        stereo = payload[:, :VIRTUAL_AUDIO_V2_CHANNELS]
    else:
        raise ValueError("audio channel count must be at least 1")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    if int(sample_rate) != VIRTUAL_AUDIO_V2_SAMPLE_RATE:
        left = resample_audio(stereo[:, 0], sample_rate=int(sample_rate), target_rate=VIRTUAL_AUDIO_V2_SAMPLE_RATE)
        right = resample_audio(stereo[:, 1], sample_rate=int(sample_rate), target_rate=VIRTUAL_AUDIO_V2_SAMPLE_RATE)
        size = min(int(left.shape[0]), int(right.shape[0]))
        stereo = np.column_stack((left[:size], right[:size]))
    return np.ascontiguousarray(stereo, dtype=np.float32)


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
    "VIRTUAL_AUDIO_PROTOCOL_V2",
    "VIRTUAL_AUDIO_V2_BIT_DEPTH",
    "VIRTUAL_AUDIO_V2_CHANNELS",
    "VIRTUAL_AUDIO_V2_DTYPE",
    "VIRTUAL_AUDIO_V2_LAYOUT",
    "VIRTUAL_AUDIO_V2_SAMPLE_RATE",
    "BridgeResponse",
    "audio_frame_count",
    "audio_peak",
    "decode_audio_packet",
    "decode_pcm16_stereo_packet",
    "encode_audio_packet",
    "encode_pcm16_stereo_packet",
]
