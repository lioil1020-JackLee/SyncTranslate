from __future__ import annotations

import numpy as np

from app.infra.audio.playback import AudioPlayback


class _FakeOutputStream:
    def __init__(self, *, device, samplerate, channels, dtype):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.started = False
        self.closed = False
        self.writes: list[np.ndarray] = []

    def start(self) -> None:
        self.started = True

    def write(self, audio: np.ndarray) -> None:
        self.writes.append(np.array(audio, copy=True))

    def abort(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_push_passthrough_writes_to_stream(monkeypatch) -> None:
    created: list[_FakeOutputStream] = []

    def _fake_output_stream(*, device, samplerate, channels, dtype):
        stream = _FakeOutputStream(device=device, samplerate=samplerate, channels=channels, dtype=dtype)
        created.append(stream)
        return stream

    monkeypatch.setattr("app.infra.audio.playback.sd.OutputStream", _fake_output_stream)
    monkeypatch.setattr(
        AudioPlayback,
        "_find_output_devices",
        staticmethod(lambda _name: [(7, {"name": "Voicemeeter Input", "max_output_channels": 2, "default_samplerate": 48000.0})]),
    )
    monkeypatch.setattr(
        AudioPlayback,
        "_resolve_supported_output_sample_rate",
        staticmethod(lambda **_kwargs: 48000.0),
    )

    playback = AudioPlayback()
    audio = np.ones((960, 1), dtype=np.float32)

    playback.push_passthrough(audio, 48000.0, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)")

    assert len(created) == 1
    assert created[0].started is True
    assert len(created[0].writes) == 1
    assert created[0].writes[0].shape == (960, 2)


def test_push_passthrough_ignores_shared_playback_volume(monkeypatch) -> None:
    created: list[_FakeOutputStream] = []

    def _fake_output_stream(*, device, samplerate, channels, dtype):
        stream = _FakeOutputStream(device=device, samplerate=samplerate, channels=channels, dtype=dtype)
        created.append(stream)
        return stream

    monkeypatch.setattr("app.infra.audio.playback.sd.OutputStream", _fake_output_stream)
    monkeypatch.setattr(
        AudioPlayback,
        "_find_output_devices",
        staticmethod(lambda _name: [(7, {"name": "Voicemeeter Input", "max_output_channels": 2, "default_samplerate": 48000.0})]),
    )
    monkeypatch.setattr(
        AudioPlayback,
        "_resolve_supported_output_sample_rate",
        staticmethod(lambda **_kwargs: 48000.0),
    )

    playback = AudioPlayback()
    playback.set_volume(0.1)
    audio = np.full((8, 1), 0.5, dtype=np.float32)

    playback.push_passthrough(audio, 48000.0, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)")

    assert len(created) == 1
    assert float(created[0].writes[0][0, 0]) == 0.5
    assert float(created[0].writes[0][0, 1]) == 0.5


def test_push_passthrough_prefers_requested_sample_rate(monkeypatch) -> None:
    created: list[_FakeOutputStream] = []
    captured: list[dict[str, object]] = []

    def _fake_output_stream(*, device, samplerate, channels, dtype):
        stream = _FakeOutputStream(device=device, samplerate=samplerate, channels=channels, dtype=dtype)
        created.append(stream)
        return stream

    def _fake_resolve_supported_output_sample_rate(**kwargs):
        captured.append(kwargs)
        return float(kwargs["requested_sample_rate"])

    monkeypatch.setattr("app.infra.audio.playback.sd.OutputStream", _fake_output_stream)
    monkeypatch.setattr(
        AudioPlayback,
        "_find_output_devices",
        staticmethod(lambda _name: [(7, {"name": "Voicemeeter Input", "max_output_channels": 2, "default_samplerate": 44100.0})]),
    )
    monkeypatch.setattr(
        AudioPlayback,
        "_resolve_supported_output_sample_rate",
        staticmethod(_fake_resolve_supported_output_sample_rate),
    )

    playback = AudioPlayback()
    audio = np.ones((960, 1), dtype=np.float32)

    playback.push_passthrough(audio, 48000.0, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)")

    assert len(created) == 1
    assert created[0].samplerate == 48000.0
    assert captured[0]["prefer_device_rate"] is False