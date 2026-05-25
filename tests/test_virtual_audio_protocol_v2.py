from __future__ import annotations

import base64

import numpy as np
import pytest

from app.infra.audio.bridge_protocol import (
    VIRTUAL_AUDIO_PROTOCOL_V2,
    VIRTUAL_AUDIO_V2_BIT_DEPTH,
    VIRTUAL_AUDIO_V2_CHANNELS,
    VIRTUAL_AUDIO_V2_DTYPE,
    VIRTUAL_AUDIO_V2_LAYOUT,
    VIRTUAL_AUDIO_V2_SAMPLE_RATE,
    decode_pcm16_stereo_packet,
    encode_pcm16_stereo_packet,
)
from app.infra.audio.bridge_process import BridgeCommandHandler, BridgeState


def test_mono_float32_encode_decode_becomes_stereo_float32() -> None:
    mono = np.linspace(-0.5, 0.5, 480, dtype=np.float32).reshape((-1, 1))

    packet = encode_pcm16_stereo_packet(mono, sample_rate=48000)
    decoded, sample_rate = decode_pcm16_stereo_packet(packet)

    assert packet["protocol_version"] == VIRTUAL_AUDIO_PROTOCOL_V2
    assert packet["sample_rate"] == VIRTUAL_AUDIO_V2_SAMPLE_RATE
    assert packet["channels"] == VIRTUAL_AUDIO_V2_CHANNELS
    assert packet["bit_depth"] == VIRTUAL_AUDIO_V2_BIT_DEPTH
    assert packet["dtype"] == VIRTUAL_AUDIO_V2_DTYPE
    assert packet["layout"] == VIRTUAL_AUDIO_V2_LAYOUT
    assert packet["frames"] == 480
    assert sample_rate == 48000
    assert decoded.dtype == np.float32
    assert decoded.shape == (480, 2)
    np.testing.assert_allclose(decoded[:, 0], decoded[:, 1], atol=1.0 / 32767.0)


def test_stereo_float32_preserves_left_right_difference() -> None:
    frames = 240
    left = np.linspace(-0.8, 0.8, frames, dtype=np.float32)
    right = np.linspace(0.8, -0.8, frames, dtype=np.float32)
    stereo = np.column_stack((left, right)).astype(np.float32)

    decoded, _sample_rate = decode_pcm16_stereo_packet(
        encode_pcm16_stereo_packet(stereo, sample_rate=48000)
    )

    assert decoded.shape == stereo.shape
    assert not np.allclose(decoded[:, 0], decoded[:, 1])
    np.testing.assert_allclose(decoded[:, 0], left, atol=1.0 / 32767.0)
    np.testing.assert_allclose(decoded[:, 1], right, atol=1.0 / 32767.0)


def test_16000_mono_resamples_to_48000_stereo() -> None:
    mono = np.ones((160,), dtype=np.float32) * 0.25

    packet = encode_pcm16_stereo_packet(mono, sample_rate=16000)
    decoded, sample_rate = decode_pcm16_stereo_packet(packet)

    assert sample_rate == 48000
    assert decoded.shape == (480, 2)
    np.testing.assert_allclose(decoded[:, 0], decoded[:, 1], atol=1.0 / 32767.0)
    assert float(np.mean(decoded)) == pytest.approx(0.25, abs=2.0 / 32767.0)


def test_wrong_protocol_version_fails() -> None:
    packet = encode_pcm16_stereo_packet(np.zeros((10, 2), dtype=np.float32), sample_rate=48000)
    packet["protocol_version"] = 1

    with pytest.raises(ValueError, match="protocol_version"):
        decode_pcm16_stereo_packet(packet)


def test_wrong_dtype_fails() -> None:
    packet = encode_pcm16_stereo_packet(np.zeros((10, 2), dtype=np.float32), sample_rate=48000)
    packet["dtype"] = "float32"

    with pytest.raises(ValueError, match="dtype"):
        decode_pcm16_stereo_packet(packet)


def test_wrong_channels_fails() -> None:
    packet = encode_pcm16_stereo_packet(np.zeros((10, 2), dtype=np.float32), sample_rate=48000)
    packet["channels"] = 1

    with pytest.raises(ValueError, match="channels"):
        decode_pcm16_stereo_packet(packet)


def test_frames_data_length_mismatch_fails() -> None:
    packet = encode_pcm16_stereo_packet(np.zeros((10, 2), dtype=np.float32), sample_rate=48000)
    packet["frames"] = 11

    with pytest.raises(ValueError, match="data length mismatch"):
        decode_pcm16_stereo_packet(packet)


def test_out_of_range_audio_clips_without_int16_overflow() -> None:
    audio = np.array([[-2.0, 2.0], [4.0, -4.0]], dtype=np.float32)

    packet = encode_pcm16_stereo_packet(audio, sample_rate=48000)
    raw = base64.b64decode(packet["data_b64"])
    pcm = np.frombuffer(raw, dtype="<i2")
    decoded, _sample_rate = decode_pcm16_stereo_packet(packet)

    assert int(np.max(pcm)) == 32767
    assert int(np.min(pcm)) == -32767
    assert float(np.max(decoded)) <= 1.0
    assert float(np.min(decoded)) >= -1.0
    np.testing.assert_allclose(decoded, np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32))


def test_decode_returns_float32() -> None:
    packet = encode_pcm16_stereo_packet(np.zeros((4, 2), dtype=np.float32), sample_rate=48000)

    decoded, _sample_rate = decode_pcm16_stereo_packet(packet)

    assert decoded.dtype == np.float32


def test_bridge_process_decode_accepts_protocol_v2_packet() -> None:
    packet = encode_pcm16_stereo_packet(np.ones((3, 1), dtype=np.float32) * 0.125, sample_rate=48000)

    decoded, sample_rate = BridgeCommandHandler._decode_packet(packet)

    assert sample_rate == 48000
    assert decoded.dtype == np.float32
    assert decoded.shape == (3, 2)


def test_bridge_process_recreates_stale_shared_memory_shape(monkeypatch) -> None:
    class _Memory:
        def __init__(self) -> None:
            self.unlinked = False

        def unlink(self) -> None:
            self.unlinked = True

    class _StaleBuffer:
        def __init__(self) -> None:
            self.memory = _Memory()
            self.closed = False

        def clear(self) -> None:
            raise ValueError("shared memory ring buffer header does not match expected shape")

        def close(self) -> None:
            self.closed = True

    class _FreshBuffer:
        pass

    stale = _StaleBuffer()
    fresh = _FreshBuffer()
    calls = {"create": 0, "attach": 0}

    def _fake_create(**_kwargs):
        calls["create"] += 1
        if calls["create"] == 1:
            raise FileExistsError("stale shared memory")
        return fresh

    def _fake_attach(**_kwargs):
        calls["attach"] += 1
        return stale

    monkeypatch.setattr("app.infra.audio.bridge_process.SharedMemoryPcmRingBuffer.create", _fake_create)
    monkeypatch.setattr("app.infra.audio.bridge_process.SharedMemoryPcmRingBuffer.attach", _fake_attach)

    buffer = BridgeCommandHandler._create_or_attach_ring_buffer(
        name="Local\\SyncTranslateVirtualMicrophoneBuffer",
        capacity_frames=48000,
        channels=2,
    )

    assert buffer is fresh
    assert calls == {"create": 2, "attach": 2}
    assert stale.memory.unlinked is True
    assert stale.closed is True


def test_bridge_process_read_remote_input_returns_protocol_v2_packet() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.cleared = False

        def snapshot(self) -> np.ndarray:
            return np.ones((4, 2), dtype=np.float32) * 0.25

        def clear(self) -> None:
            self.cleared = True

    class _Event:
        def __init__(self) -> None:
            self.reset_called = False

        def wait(self, _timeout: float) -> bool:
            return True

        def reset(self) -> None:
            self.reset_called = True

    handler = object.__new__(BridgeCommandHandler)
    handler.state = BridgeState(remote_input_running=False, remote_input_sample_rate=48000)
    handler.remote_input_buffer = _Buffer()
    handler.remote_input_event = _Event()

    response = handler.handle({"cmd": "read_remote_input", "sample_rate": 48000})

    assert response["ok"] is True
    packet = response["packet"]
    assert packet["protocol_version"] == 2
    decoded, sample_rate = decode_pcm16_stereo_packet(packet)
    assert sample_rate == 48000
    assert decoded.shape == (4, 2)
    assert handler.remote_input_buffer.cleared is True
    assert handler.remote_input_event.reset_called is True


def test_bridge_process_virtual_mic_write_decodes_protocol_v2_packet() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.written: np.ndarray | None = None

        def write(self, audio: np.ndarray) -> int:
            self.written = audio.copy()
            return int(audio.shape[0])

    class _Event:
        def __init__(self) -> None:
            self.set_called = False

        def set(self) -> None:
            self.set_called = True

    handler = object.__new__(BridgeCommandHandler)
    handler.state = BridgeState()
    handler.virtual_microphone_buffer = _Buffer()
    handler.virtual_microphone_event = _Event()
    packet = encode_pcm16_stereo_packet(np.ones((6, 1), dtype=np.float32) * 0.2, sample_rate=48000)

    response = handler.handle({"cmd": "write_virtual_microphone", "packet": packet})

    assert response["ok"] is True
    assert response["written_frames"] == 6
    assert handler.virtual_microphone_buffer.written is not None
    assert handler.virtual_microphone_buffer.written.shape == (6, 2)
    assert handler.virtual_microphone_event.set_called is True
