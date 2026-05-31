from __future__ import annotations

import time

import numpy as np

from app.infra.audio.loopback_capture import WasapiLoopbackCaptureSource


class _Recorder:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def record(self, numframes: int):
        time.sleep(0.005)
        return np.ones((int(numframes), 2), dtype=np.float32) * 0.05


class _Microphone:
    def recorder(self, *, samplerate: float, channels: int, blocksize: int, exclusive_mode: bool = False):
        return _Recorder()


def test_loopback_capture_falls_back_to_soundcard_backend(monkeypatch) -> None:
    source = WasapiLoopbackCaptureSource()
    chunks: list[np.ndarray] = []

    monkeypatch.setattr(source, "_can_use_sounddevice_loopback", lambda: False)
    monkeypatch.setattr(source, "_resolve_output_device", lambda device: (1, {"max_output_channels": 2, "default_samplerate": 48000.0}))
    monkeypatch.setattr(source, "_resolve_soundcard_loopback_microphone", lambda device: _Microphone())
    monkeypatch.setattr("app.infra.audio.loopback_capture.sc", object())
    monkeypatch.setattr(source, "_install_soundcard_numpy_compat", lambda: None)
    monkeypatch.setattr(source, "_initialize_com_for_thread", lambda: False)

    source.add_consumer(lambda audio, rate: chunks.append(audio.copy()))
    source.start("Speakers", sample_rate=48000, chunk_ms=10)
    time.sleep(0.04)
    source.stop()

    stats = source.stats()
    assert chunks
    assert chunks[0].dtype == np.float32
    assert chunks[0].shape[1] == 2
    assert stats.frame_count > 0
    assert stats.channels == 2
