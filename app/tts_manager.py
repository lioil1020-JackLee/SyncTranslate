from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable

from app.audio_playback import AudioPlayback
from app.local_ai.tts_factory import create_tts_engine
from app.schemas import AppConfig, TtsConfig, merge_tts_configs


@dataclass(slots=True)
class _TtsTask:
    channel: str
    text: str


class TTSManager:
    def __init__(
        self,
        *,
        config: AppConfig,
        local_playback: AudioPlayback,
        remote_playback: AudioPlayback,
        get_local_output_device: Callable[[], str],
        get_remote_output_device: Callable[[], str],
        on_error: Callable[[str], None] | None = None,
        on_play_start: Callable[[str], None] | None = None,
        on_play_end: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._playbacks = {
            "local": local_playback,
            "remote": remote_playback,
        }
        self._output_getters = {
            "local": get_local_output_device,
            "remote": get_remote_output_device,
        }
        self._on_error = on_error
        self._on_play_start = on_play_start
        self._on_play_end = on_play_end
        self._queue: Queue[_TtsTask] = Queue(maxsize=max(8, self._config.runtime.tts_queue_maxsize))
        self._stop_event = Event()
        self._worker: Thread | None = None

    def set_callbacks(
        self,
        *,
        on_play_start: Callable[[str], None] | None = None,
        on_play_end: Callable[[str], None] | None = None,
    ) -> None:
        self._on_play_start = on_play_start
        self._on_play_end = on_play_end

    def start(self) -> None:
        if not self.stop(wait_timeout=5.0):
            raise RuntimeError("TTS worker is still shutting down")
        self._stop_event.clear()
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self, wait_timeout: float = 5.0) -> bool:
        self._stop_event.set()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=max(0.1, float(wait_timeout)))
        stopped = worker is None or not worker.is_alive()
        if stopped:
            self._worker = None
        for playback in self._playbacks.values():
            playback.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        return stopped

    def enqueue(self, channel: str, text: str) -> None:
        key = channel if channel in ("local", "remote") else "local"
        if not text.strip():
            return
        try:
            self._queue.put_nowait(_TtsTask(channel=key, text=text.strip()))
        except Exception:
            if self._on_error:
                self._on_error(f"tts_{key} queue overflow")

    def set_volume(self, channel: str, volume: float) -> None:
        key = channel if channel in ("local", "remote") else "local"
        self._playbacks[key].set_volume(volume)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except Empty:
                continue

            if self._stop_event.is_set():
                break

            channel = task.channel
            tts_cfg = self._channel_tts_config(channel)
            playback = self._playbacks[channel]
            output_device = self._output_getters[channel]()
            try:
                if self._on_play_start:
                    self._on_play_start(channel)
                engine = create_tts_engine(tts_cfg)
                audio = engine.synthesize(task.text, sample_rate=tts_cfg.sample_rate)
                if self._stop_event.is_set():
                    break
                playback.play(
                    audio=audio,
                    sample_rate=tts_cfg.sample_rate,
                    output_device_name=output_device,
                )
            except Exception as exc:
                if self._on_error:
                    self._on_error(f"tts_{channel} playback failed: {exc}")
            finally:
                if self._on_play_end:
                    self._on_play_end(channel)

    def _channel_tts_config(self, channel: str) -> TtsConfig:
        if channel == "remote":
            fallback = self._config.local_tts
            override = self._config.tts_channels.remote
        else:
            fallback = self._config.meeting_tts
            override = self._config.tts_channels.local
        return merge_tts_configs(self._config.tts, fallback, override)
