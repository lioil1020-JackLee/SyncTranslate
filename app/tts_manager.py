from __future__ import annotations

from dataclasses import dataclass
import time
from collections import deque
from threading import Condition, Event, Thread
from typing import Any, Callable

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
        self._maxsize_by_channel = {
            "local": max(
                8,
                int(getattr(self._config.runtime, "tts_queue_maxsize_local", self._config.runtime.tts_queue_maxsize)),
            ),
            "remote": max(
                8,
                int(getattr(self._config.runtime, "tts_queue_maxsize_remote", self._config.runtime.tts_queue_maxsize)),
            ),
        }
        self._drop_threshold = max(2, int(self._config.runtime.tts_drop_backlog_threshold))
        self._cancel_pending_on_new_final = bool(self._config.runtime.tts_cancel_pending_on_new_final)
        self._cancel_policy = str(getattr(self._config.runtime, "tts_cancel_policy", "all_pending") or "all_pending")
        self._max_wait_ms = max(500, int(getattr(self._config.runtime, "tts_max_wait_ms", 4000)))
        self._max_chars = max(20, int(getattr(self._config.runtime, "tts_max_chars", 200)))
        self._pending: deque[_TtsTask] = deque()
        self._queue_changed = Condition()
        self._drop_count = {"local": 0, "remote": 0}
        self._engine_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
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
        self._engine_cache.clear()
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
        chunks = self._split_text(cleaned, self._max_chars)
        now = time.time()
        base_revision = max(0, int(revision))
        tasks = [
            _TtsTask(
                channel=key,
                text=chunk,
                utterance_id=utterance_id,
                revision=base_revision + idx,
                created_at=now,
            )
            for idx, chunk in enumerate(chunks)
            if chunk.strip()
        ]
        if not tasks:
            return
        dropped = 0
        with self._queue_changed:
            if self._cancel_pending_on_new_final:
                if self._cancel_policy == "older_only":
                    dropped += self._drop_older_for_channel_locked(key, utterance_id, base_revision)
                else:
                    dropped += self._drop_pending_for_channel_locked(key)
            for task in tasks:
                self._pending.append(task)
                dropped += self._shed_backlog_locked(key)
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
                engine = self._resolve_engine(channel, tts_cfg)
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
            while self._pending:
                task = self._pending.popleft()
                age_ms = max(0.0, (time.time() - task.created_at) * 1000.0)
                if age_ms <= self._max_wait_ms:
                    return task
                self._drop_count[task.channel] = self._drop_count.get(task.channel, 0) + 1
                if self._on_error:
                    self._on_error(
                        ErrorEvent(
                            level="warning",
                            module="tts_manager",
                            source=task.channel,
                            code="queue_timeout_drop",
                            message="Dropped expired TTS task",
                            detail=f"wait_ms={int(age_ms)} limit_ms={self._max_wait_ms}",
                        )
                    )
            return None

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

    def _drop_older_for_channel_locked(self, channel: str, utterance_id: str | None, revision: int) -> int:
        if not self._pending:
            return 0
        kept: deque[_TtsTask] = deque()
        dropped = 0
        for task in self._pending:
            if task.channel != channel:
                kept.append(task)
                continue
            is_same_utterance = bool(utterance_id) and task.utterance_id == utterance_id
            if is_same_utterance and task.revision >= revision:
                kept.append(task)
                continue
            dropped += 1
        self._pending = kept
        return dropped

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        source = (text or "").strip()
        if not source:
            return []
        if len(source) <= max_chars:
            return [source]
        chunks: list[str] = []
        rest = source
        delimiters = "。！？!?；;，,"
        while len(rest) > max_chars:
            candidate = rest[:max_chars]
            cut = -1
            for i in range(len(candidate) - 1, -1, -1):
                if candidate[i] in delimiters:
                    cut = i + 1
                    break
            if cut <= 0:
                cut = candidate.rfind(" ")
                if cut <= 0:
                    cut = max_chars
            chunk = rest[:cut].strip()
            if chunk:
                chunks.append(chunk)
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
        return chunks

    def _shed_backlog_locked(self, channel: str) -> int:
        dropped = 0
        limit = min(self._drop_threshold, self._maxsize_by_channel.get(channel, self._drop_threshold))
        while sum(1 for task in self._pending if task.channel == channel) > limit:
            removed = False
            for idx, task in enumerate(self._pending):
                if task.channel == channel:
                    del self._pending[idx]
                    removed = True
                    dropped += 1
                    break
            if not removed:
                break
        return dropped

    def _channel_tts_config(self, channel: str) -> TtsConfig:
        if channel == "remote":
            fallback = self._config.local_tts
            override = self._config.tts_channels.remote
        else:
            fallback = self._config.meeting_tts
            override = self._config.tts_channels.local
        return merge_tts_configs(self._config.tts, fallback, override)

    def _resolve_engine(self, channel: str, tts_cfg: TtsConfig) -> Any:
        config_key: tuple[Any, ...] = (
            tts_cfg.engine,
            tts_cfg.executable_path,
            tts_cfg.model_path,
            tts_cfg.config_path,
            tts_cfg.voice_name,
            tts_cfg.speaker_id,
            tts_cfg.length_scale,
            tts_cfg.noise_scale,
            tts_cfg.noise_w,
            tts_cfg.sample_rate,
        )
        cached = self._engine_cache.get(channel)
        if cached and cached[0] == config_key:
            return cached[1]
        engine = create_tts_engine(tts_cfg)
        self._engine_cache[channel] = (config_key, engine)
        return engine
