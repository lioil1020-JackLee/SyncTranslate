from __future__ import annotations

import ctypes
import inspect
import time
import warnings
from threading import Event, Thread
from typing import Callable

import numpy as np
import sounddevice as sd

from app.infra.audio.capture import AudioCapture, CaptureStats
from app.infra.audio.device_registry import (
    device_tokens,
    list_indexed_devices,
    normalize_device_text,
    parse_device_selector,
    preferred_hostapi_index_for_platform,
)
from app.infra.audio.frame import ChannelPolicy

try:
    import soundcard as sc
except Exception:
    sc = None


class WasapiLoopbackCaptureSource:
    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._capture_stats = CaptureStats(False, False, 0.0, 0, 0.0, "")
        self._channels = 0
        self._consumers: list[Callable[[np.ndarray, float], None]] = []

    def start(
        self,
        device_name: str,
        sample_rate: int,
        chunk_ms: int,
        channels_policy: str = ChannelPolicy.STEREO_OR_MONO.value,
    ) -> None:
        self.stop()
        device_index, device_info = self._resolve_output_device(device_name)
        max_output_channels = int(device_info.get("max_output_channels", 0) or 0)
        channels = 2 if max_output_channels >= 2 else 1
        if str(channels_policy).lower() == ChannelPolicy.MONO.value:
            channels = 1
        rate = float(sample_rate or device_info.get("default_samplerate") or 48000)
        blocksize = max(256, int(round(rate * max(5, int(chunk_ms)) / 1000.0)))
        if self._can_use_sounddevice_loopback():
            extra_settings = sd.WasapiSettings(loopback=True)
            self._stream = sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=rate,
                blocksize=blocksize,
                dtype="float32",
                extra_settings=extra_settings,
                callback=self._on_audio,
            )
            self._stream.start()
        else:
            self._start_soundcard_loopback(device_name, rate=rate, channels=channels, blocksize=blocksize)
        self._channels = int(channels)
        self._capture_stats = CaptureStats(True, False, rate, 0, 0.0, "", channels=int(channels))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.5)
            self._thread = None
        self._capture_stats = CaptureStats(False, False, self._capture_stats.sample_rate, self._capture_stats.frame_count, self._capture_stats.level, self._capture_stats.last_error, channels=self._channels)

    def add_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def remove_consumer(self, consumer: Callable[[np.ndarray, float], None]) -> None:
        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def set_gain(self, gain: float) -> None:
        del gain

    def stats(self) -> CaptureStats:
        return self._capture_stats

    def _on_audio(self, indata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        audio = np.asarray(indata, dtype=np.float32).copy()
        level = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        self._capture_stats = CaptureStats(
            True,
            False,
            self._capture_stats.sample_rate,
            int(self._capture_stats.frame_count) + int(frames),
            level,
            str(status) if status else "",
            channels=int(audio.shape[1]) if audio.ndim == 2 else 1,
        )
        for consumer in list(self._consumers):
            consumer(audio, self._capture_stats.sample_rate)

    def _start_soundcard_loopback(self, device_name: str, *, rate: float, channels: int, blocksize: int) -> None:
        if sc is None:
            raise RuntimeError(
                "WASAPI loopback is unavailable: sounddevice has no loopback setting and soundcard is not installed"
            )
        microphone = self._resolve_soundcard_loopback_microphone(device_name)
        startup_ready = Event()
        startup_errors: list[str] = []
        self._stop_event.clear()

        def _worker() -> None:
            com_initialized = self._initialize_com_for_thread()
            try:
                self._install_soundcard_numpy_compat()
                with microphone.recorder(
                    samplerate=float(rate),
                    channels=int(channels),
                    blocksize=int(blocksize),
                ) as recorder:
                    startup_ready.set()
                    next_read_at = time.perf_counter()
                    block_duration_sec = max(0.001, float(blocksize) / float(rate or 48000.0))
                    while not self._stop_event.is_set():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="data discontinuity in recording",
                                category=RuntimeWarning,
                            )
                            audio = np.asarray(recorder.record(numframes=int(blocksize)), dtype=np.float32)
                        if audio.ndim == 1:
                            audio = audio.reshape((-1, 1))
                        self._on_audio(audio, int(audio.shape[0]), None, "")
                        next_read_at += block_duration_sec
                        delay = next_read_at - time.perf_counter()
                        if delay > 0:
                            self._stop_event.wait(delay)
                        elif delay < -block_duration_sec:
                            next_read_at = time.perf_counter()
            except Exception as exc:
                startup_errors.append(str(exc))
                self._capture_stats = CaptureStats(
                    False,
                    False,
                    float(rate),
                    int(self._capture_stats.frame_count),
                    float(self._capture_stats.level),
                    str(exc),
                    channels=int(channels),
                )
                startup_ready.set()
            finally:
                if com_initialized:
                    try:
                        ctypes.OleDLL("ole32").CoUninitialize()
                    except Exception:
                        pass

        self._thread = Thread(target=_worker, name="wasapi-loopback-capture", daemon=True)
        self._thread.start()
        if not startup_ready.wait(timeout=3.0):
            self.stop()
            raise RuntimeError(f"Output loopback capture did not start within 3 seconds: {device_name}")
        if startup_errors:
            self.stop()
            raise RuntimeError(f"Output loopback capture failed: {startup_errors[0]}")

    @staticmethod
    def _can_use_sounddevice_loopback() -> bool:
        settings = getattr(sd, "WasapiSettings", None)
        if settings is None:
            return False
        try:
            return "loopback" in inspect.signature(settings).parameters
        except Exception:
            return False

    @staticmethod
    def _resolve_soundcard_loopback_microphone(device_name: str):
        if sc is None:
            raise RuntimeError("soundcard_unavailable")
        target = parse_device_selector(device_name)[1]
        normalized_target = normalize_device_text(target)
        target_tokens = device_tokens(target)
        ranked: list[tuple[int, int, int, object]] = []
        for index, microphone in enumerate(sc.all_microphones(include_loopback=True)):
            if not bool(getattr(microphone, "isloopback", False)):
                continue
            name = str(getattr(microphone, "name", "") or "")
            normalized_name = normalize_device_text(name)
            name_tokens = device_tokens(name)
            score = 0
            if name == target:
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
                ranked.append((-score, extra_token_penalty, index, microphone))
        if not ranked:
            raise ValueError(f"Output loopback device not found by soundcard backend: {device_name}")
        ranked.sort()
        return ranked[0][3]

    @staticmethod
    def _install_soundcard_numpy_compat() -> None:
        if getattr(np.fromstring, "_synctranslate_soundcard_compat", False):
            return
        original_fromstring = np.fromstring

        def _fromstring_compat(data, dtype=float, count=-1, sep="", *, like=None):
            if sep:
                return original_fromstring(data, dtype=dtype, count=count, sep=sep, like=like)
            return np.frombuffer(data, dtype=dtype, count=count, like=like)

        _fromstring_compat._synctranslate_soundcard_compat = True  # type: ignore[attr-defined]
        np.fromstring = _fromstring_compat  # type: ignore[assignment]

    @staticmethod
    def _initialize_com_for_thread() -> bool:
        try:
            ole32 = ctypes.OleDLL("ole32")
            hr = ole32.CoInitializeEx(None, 0)
            if hr in (0, 1):
                return True
            if hr in (-2147417850, 0x80010106):
                return False
        except Exception:
            return False
        return False

    @staticmethod
    def _resolve_output_device(device_name: str) -> tuple[int, dict[str, object]]:
        hostapi_name, requested_name = parse_device_selector(device_name)
        preferred_hostapi = preferred_hostapi_index_for_platform()
        normalized_target = normalize_device_text(requested_name)
        target_tokens = device_tokens(requested_name)
        ranked: list[tuple[int, int, int, dict[str, object]]] = []
        for idx, item in list_indexed_devices():
            if int(item.get("max_output_channels", 0) or 0) <= 0:
                continue
            item_name = str(item.get("name", ""))
            normalized_name = normalize_device_text(item_name)
            name_tokens = device_tokens(item_name)
            score = 0
            if item_name == requested_name:
                score = 500
            elif normalized_name == normalized_target:
                score = 450
            elif normalized_target and normalized_target in normalized_name:
                score = 350
            elif target_tokens and target_tokens.issubset(name_tokens):
                score = 300
            if score <= 0:
                continue
            item_hostapi = int(item.get("hostapi", -1))
            hostapi_rank = 0 if (hostapi_name and str(sd.query_hostapis()[item_hostapi].get("name", "")) == hostapi_name) or (not hostapi_name and item_hostapi == preferred_hostapi) else 1
            ranked.append((hostapi_rank, -score, idx, item))
        if not ranked:
            raise ValueError(f"Output loopback device not found or unavailable: {device_name}")
        ranked.sort()
        _, _, idx, item = ranked[0]
        return idx, item


__all__ = ["WasapiLoopbackCaptureSource"]
