from __future__ import annotations

from dataclasses import dataclass
import time
from collections import deque
from threading import Condition, Event, Thread
from typing import Callable

from app.audio_playback import AudioPlayback
from app.events import ErrorEvent
from app.local_ai.tts_factory import create_tts_engine
from app.schemas import AppConfig, TtsConfig, merge_tts_configs


@dataclass(slots=True)
class _TtsTask:
    channel: str
    text: str
    utterance_id: str | None
    revision: int
    created_at: float


class TTSManager:
    def __init__(
        self,
        *,
        config: AppConfig,
        local_playback: AudioPlayback,
        remote_playback: AudioPlayback,
        get_local_output_device: Callable[[], str],
        get_remote_output_device: Callable[[], str],
        on_error: Callable[[str | ErrorEvent], None] | None = None,
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
        self._maxsize = max(8, self._config.runtime.tts_queue_maxsize)
        self._drop_threshold = max(2, int(self._config.runtime.tts_drop_backlog_threshold))
        self._cancel_pending_on_new_final = bool(self._config.runtime.tts_cancel_pending_on_new_final)
        self._pending: deque[_TtsTask] = deque()
        self._queue_changed = Condition()
        self._drop_count = {"local": 0, "remote": 0}
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
        with self._queue_changed:
            self._pending.clear()
            self._queue_changed.notify_all()
        return stopped

    def enqueue(
        self,
        channel: str,
        text: str,
        *,
        utterance_id: str | None = None,
        revision: int = 0,
        is_final: bool = True,
    ) -> None:
        key = channel if channel in ("local", "remote") else "local"
        if not is_final:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        task = _TtsTask(
            channel=key,
            text=cleaned,
            utterance_id=utterance_id,
            revision=max(0, int(revision)),
            created_at=time.time(),
        )
        dropped = 0
        with self._queue_changed:
            if self._cancel_pending_on_new_final:
                dropped += self._drop_pending_for_channel_locked(key)
            self._pending.append(task)
            dropped += self._shed_backlog_locked()
            self._queue_changed.notify()
        if dropped > 0:
            self._drop_count[key] += dropped
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="warning",
                        module="tts_manager",
                        source=key,
                        code="queue_shed",
                        message="Dropped stale TTS tasks",
                        detail=f"count={dropped}",
                    )
                )

    def set_volume(self, channel: str, volume: float) -> None:
        key = channel if channel in ("local", "remote") else "local"
        self._playbacks[key].set_volume(volume)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            task = self._next_task()
            if task is None:
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
                    self._on_error(
                        ErrorEvent(
                            level="error",
                            module="tts_manager",
                            source=channel,
                            code="playback_failed",
                            message="TTS playback failed",
                            detail=str(exc),
                        )
                    )
            finally:
                if self._on_play_end:
                    self._on_play_end(channel)

    def stats(self) -> dict[str, object]:
        with self._queue_changed:
            depth_by_channel = {
                "local": sum(1 for task in self._pending if task.channel == "local"),
                "remote": sum(1 for task in self._pending if task.channel == "remote"),
            }
            oldest_age_ms = 0.0
            if self._pending:
                oldest_age_ms = max(0.0, (time.time() - self._pending[0].created_at) * 1000.0)
            return {
                "queue_depth": len(self._pending),
                "queue_depth_local": depth_by_channel["local"],
                "queue_depth_remote": depth_by_channel["remote"],
                "drop_count_local": self._drop_count["local"],
                "drop_count_remote": self._drop_count["remote"],
                "oldest_age_ms": oldest_age_ms,
            }

    def _next_task(self) -> _TtsTask | None:
        with self._queue_changed:
            while not self._stop_event.is_set() and not self._pending:
                self._queue_changed.wait(timeout=0.2)
            if self._stop_event.is_set():
                return None
            return self._pending.popleft() if self._pending else None

    def _drop_pending_for_channel_locked(self, channel: str) -> int:
        if not self._pending:
            return 0
        kept: deque[_TtsTask] = deque()
        dropped = 0
        while self._pending:
            task = self._pending.popleft()
            if task.channel == channel:
                dropped += 1
                continue
            kept.append(task)
        self._pending = kept
        return dropped

    def _shed_backlog_locked(self) -> int:
        dropped = 0
        limit = min(self._drop_threshold, self._maxsize)
        while len(self._pending) > limit:
            self._pending.popleft()
            dropped += 1
        return dropped

    def _channel_tts_config(self, channel: str) -> TtsConfig:
        if channel == "remote":
            fallback = self._config.local_tts
            override = self._config.tts_channels.remote
        else:
            fallback = self._config.meeting_tts
            override = self._config.tts_channels.local
        return merge_tts_configs(self._config.tts, fallback, override)
