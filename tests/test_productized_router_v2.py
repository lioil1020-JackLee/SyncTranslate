from __future__ import annotations

import numpy as np

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.config.schema import AppConfig


class _InputManager:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, str]] = []
        self.consumers = {}

    def add_consumer(self, source, consumer) -> None:
        self.consumers[source] = consumer

    def remove_consumer(self, source, consumer) -> None:
        self.consumers.pop(source, None)

    def start(self, source, device_name, sample_rate, chunk_ms, channels_policy="mono") -> None:
        self.started.append((source, device_name, channels_policy))

    def stop(self, source) -> None:
        pass

    def stop_all(self) -> None:
        pass

    def stats(self):
        return {}


class _Asr:
    def __init__(self) -> None:
        self.submits = []
        self.started = []

    def set_enabled(self, source, enabled) -> None:
        pass

    def start(self, source, callback) -> None:
        self.started.append(source)

    def stop(self, source) -> None:
        pass

    def stop_all(self) -> None:
        pass

    def configure_pipeline(self, config, revision) -> None:
        pass

    def submit(self, source, chunk, sample_rate) -> None:
        self.submits.append((source, chunk, sample_rate))

    def stats(self):
        return {}


class _Translator:
    def update_config(self, config) -> None:
        pass


class _Tts:
    def __init__(self) -> None:
        self.direct = []
        self.modes = {"local": "subtitle_only", "remote": "subtitle_only"}

    def start(self) -> None:
        pass

    def stop(self, wait_timeout=1.5) -> bool:
        return True

    def set_output_mode(self, channel, mode) -> None:
        self.modes[channel] = mode

    def output_mode(self, channel):
        return self.modes.get(channel, "subtitle_only")

    def is_passthrough_enabled(self, channel):
        return self.modes.get(channel) == "passthrough"

    def submit_passthrough(self, channel, chunk, sample_rate) -> None:
        raise AssertionError("legacy passthrough should not be used")

    def submit_direct_passthrough(self, channel, chunk, sample_rate) -> None:
        self.direct.append((channel, chunk, sample_rate))

    def stats(self):
        return {}


def _router(config: AppConfig):
    asr = _Asr()
    tts = _Tts()
    inputs = _InputManager()
    router = AudioRouter(
        transcript_buffer=TranscriptBuffer(),
        input_manager=inputs,
        asr_manager=asr,
        translator_manager=_Translator(),
        tts_manager=tts,
        state_manager=StateManager(),
    )
    router.refresh_runtime_config(config)
    return router, inputs, asr, tts


def test_meeting_mode_starts_only_monitor_source_and_keeps_asr_on_with_tts_none() -> None:
    cfg = AppConfig.from_dict({"runtime": {"session_mode": "meeting"}, "meeting": {"input_device": "Mic"}})
    router, inputs, asr, _tts = _router(cfg)

    router.start("meeting", cfg.audio, sample_rate=48000, chunk_ms=40)

    assert inputs.started == [("remote", "Mic", "mono")]
    assert asr.started == ["remote"]
    inputs.consumers["remote"](np.ones((480, 1), dtype=np.float32), 48000.0)
    assert len(asr.submits) == 1
    assert asr.submits[0][1].ndim == 1
    assert asr.submits[0][2] == 16000.0


def test_dialogue_voice_none_uses_direct_passthrough_and_skips_asr() -> None:
    cfg = AppConfig.from_dict(
        {
            "runtime": {"session_mode": "dialogue"},
            "dialogue": {
                "remote_to_local": {"tts_voice": "none"},
                "local_to_remote": {"tts_voice": "none"},
            },
            "audio": {"meeting_in": "Virtual Speaker", "microphone_in": "Mic"},
        }
    )
    router, inputs, asr, tts = _router(cfg)

    router.start("dialogue", cfg.audio, sample_rate=48000, chunk_ms=40)

    assert ("remote", "Virtual Speaker", "stereo_or_mono") in inputs.started
    assert ("local", "Mic", "stereo_or_mono") in inputs.started
    inputs.consumers["remote"](np.ones((480, 2), dtype=np.float32), 48000.0)
    assert asr.submits == []
    assert len(tts.direct) == 1
    assert tts.direct[0][0] == "local"
