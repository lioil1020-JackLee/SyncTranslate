from __future__ import annotations

from dataclasses import dataclass
import time
from collections import deque
from threading import Condition, Event, Thread
from typing import Any, Callable

import numpy as np

from app.domain.events import ErrorEvent
from app.infra.audio.playback import AudioPlayback
from app.infra.tts.engine import create_tts_engine
from app.infra.tts.voice_policy import resolve_tts_config_for_target
from app.infra.config.schema import AppConfig, TtsConfig


@dataclass(slots=True)
class _TtsTask:
    channel: str
    text: str
    utterance_id: str | None
    revision: int
    created_at: float
    is_final: bool
    is_stable_partial: bool
    is_early_final: bool


@dataclass(slots=True)
class _PassthroughTask:
    audio: np.ndarray
    sample_rate: float
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
                4,
                int(getattr(self._config.runtime, "tts_queue_maxsize_local", self._config.runtime.tts_queue_maxsize)),
            ),
            "remote": max(
                4,
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
        initial_mode = str(getattr(self._config.runtime, "tts_output_mode", "subtitle_only") or "subtitle_only")
        if initial_mode not in {"tts", "subtitle_only", "passthrough"}:
            initial_mode = "subtitle_only"
        self._output_mode: dict[str, str] = {"local": initial_mode, "remote": initial_mode}
        self._passthrough_pending: dict[str, deque[_PassthroughTask]] = {
            "local": deque(),
            "remote": deque(),
        }
        # Keep a short jitter buffer to avoid robotic audio caused by frequent tiny chunk drops.
        self._passthrough_queue_limit = 24
        self._passthrough_warmup_sec = 0.24
        self._passthrough_warmup_until: dict[str, float] = {"local": 0.0, "remote": 0.0}
        self._passthrough_gate_timeout_sec = 0.60
        self._passthrough_gate_rms = 0.0055
        self._passthrough_gate_until: dict[str, float] = {"local": 0.0, "remote": 0.0}
        self._passthrough_drop_count = {"local": 0, "remote": 0}
        self._passthrough_stop = Event()
        self._passthrough_workers: dict[str, Thread | None] = {"local": None, "remote": None}
        self._engine_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._current_task_by_channel: dict[str, _TtsTask | None] = {"local": None, "remote": None}
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
        self._passthrough_stop.clear()
        for channel in ("local", "remote"):
            worker = Thread(target=self._run_passthrough, args=(channel,), daemon=True)
            self._passthrough_workers[channel] = worker
            worker.start()

    def stop(self, wait_timeout: float = 5.0) -> bool:
        self._stop_event.set()
        self._passthrough_stop.set()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=max(0.1, float(wait_timeout)))
        stopped = worker is None or not worker.is_alive()
        if stopped:
            self._worker = None
        for channel in ("local", "remote"):
            passthrough_worker = self._passthrough_workers.get(channel)
            if passthrough_worker and passthrough_worker.is_alive():
                passthrough_worker.join(timeout=max(0.1, float(wait_timeout)))
            self._passthrough_workers[channel] = None
        for playback in self._playbacks.values():
            playback.stop()
        with self._queue_changed:
            self._pending.clear()
            self._passthrough_pending["local"].clear()
            self._passthrough_pending["remote"].clear()
            self._queue_changed.notify_all()
        self._engine_cache.clear()
        self._current_task_by_channel = {"local": None, "remote": None}
        return stopped

    def enqueue(
        self,
        channel: str,
        text: str,
        *,
        utterance_id: str | None = None,
        revision: int = 0,
        is_final: bool = True,
        is_stable_partial: bool = False,
        is_early_final: bool = False,
    ) -> None:
        key = channel if channel in ("local", "remote") else "local"
        if self._output_mode.get(key) != "tts":
            return
        accept_stable_partial = bool(getattr(self._config.runtime, "tts_accept_stable_partial", True))
        if not is_final and not (accept_stable_partial and is_stable_partial):
            return
        cleaned = text.strip()
        if not cleaned:
            return
        partial_min_chars = max(1, int(getattr(self._config.runtime, "tts_partial_min_chars", 12)))
        if not is_final and len(cleaned) < partial_min_chars:
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
                is_final=is_final,
                is_stable_partial=bool(is_stable_partial and not is_final),
                is_early_final=bool(is_early_final),
            )
            for idx, chunk in enumerate(chunks)
            if chunk.strip()
        ]
        if not tasks:
            return
        dropped = 0
        with self._queue_changed:
            if self._cancel_pending_on_new_final or is_early_final or is_stable_partial:
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

    def set_muted(self, channel: str, muted: bool) -> None:
        key = channel if channel in ("local", "remote") else "local"
        self.set_output_mode(key, "subtitle_only" if muted else "tts")

    def set_passthrough_enabled(self, channel: str, enabled: bool) -> None:
        key = channel if channel in ("local", "remote") else "local"
        self.set_output_mode(key, "passthrough" if enabled else "tts")

    def set_output_mode(self, channel: str, mode: str) -> None:
        key = channel if channel in ("local", "remote") else "local"
        normalized = mode if mode in {"tts", "subtitle_only", "passthrough"} else "subtitle_only"
        self._output_mode[key] = normalized
        self._playbacks[key].stop()
        with self._queue_changed:
            self._pending = deque(task for task in self._pending if task.channel != key)
            self._passthrough_pending[key].clear()
            self._queue_changed.notify_all()
        if normalized == "passthrough":
            self._passthrough_warmup_until[key] = time.time() + self._passthrough_warmup_sec
            self._passthrough_gate_until[key] = time.time() + self._passthrough_gate_timeout_sec
        else:
            self._passthrough_warmup_until[key] = 0.0
            self._passthrough_gate_until[key] = 0.0

    def is_passthrough_enabled(self, channel: str) -> bool:
        key = channel if channel in ("local", "remote") else "local"
        return self._output_mode.get(key) == "passthrough"

    def output_mode(self, channel: str) -> str:
        key = channel if channel in ("local", "remote") else "local"
        return self._output_mode.get(key, "subtitle_only")

    def submit_passthrough(self, channel: str, audio: np.ndarray, sample_rate: float) -> None:
        key = channel if channel in ("local", "remote") else "local"
        if self._output_mode.get(key) != "passthrough":
            return
        now = time.time()
        if now < self._passthrough_warmup_until.get(key, 0.0):
            return
        if audio.size == 0 or sample_rate <= 0:
            return
        try:
            passthrough_audio = audio.astype(np.float32, copy=False)
            gain = max(0.1, float(getattr(self._config.runtime, "passthrough_gain", 1.6) or 1.6))
            if gain != 1.0:
                passthrough_audio = np.clip(passthrough_audio * gain, -1.0, 1.0).astype(np.float32, copy=False)
            self._playbacks[key].push_passthrough(
                audio=passthrough_audio,
                sample_rate=float(sample_rate),
                output_device_name=self._output_getters[key](),
            )
        except Exception as exc:
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="warning",
                        module="tts_manager",
                        source=key,
                        code="passthrough_stream_failed",
                        message="Passthrough streaming failed",
                        detail=str(exc),
                    )
                )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            task = self._next_task()
            if task is None:
                continue

            if self._stop_event.is_set():
                break

            channel = task.channel
            if self._output_mode.get(channel) != "tts":
                # Skip silently — on_play_start was never called for this task,
                # so do NOT call on_play_end either (avoids spurious StateManager echo-guard triggers).
                continue
            tts_cfg = self._channel_tts_config(channel)
            playback = self._playbacks[channel]
            output_device = self._output_getters[channel]()
            try:
                self._current_task_by_channel[channel] = task
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
                self._current_task_by_channel[channel] = None
                if self._on_play_end:
                    self._on_play_end(channel)

    def stats(self) -> dict[str, object]:
        with self._queue_changed:
            depth_by_channel = {
                "local": sum(1 for task in self._pending if task.channel == "local"),
                "remote": sum(1 for task in self._pending if task.channel == "remote"),
            }
            passthrough_depth = {
                "local": len(self._passthrough_pending["local"]),
                "remote": len(self._passthrough_pending["remote"]),
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
                "output_mode_local": self._output_mode["local"],
                "output_mode_remote": self._output_mode["remote"],
                "passthrough_enabled_local": self._output_mode["local"] == "passthrough",
                "passthrough_enabled_remote": self._output_mode["remote"] == "passthrough",
                "passthrough_depth_local": passthrough_depth["local"],
                "passthrough_depth_remote": passthrough_depth["remote"],
                "passthrough_drop_count_local": self._passthrough_drop_count["local"],
                "passthrough_drop_count_remote": self._passthrough_drop_count["remote"],
                "oldest_age_ms": oldest_age_ms,
            }

    def current_task(self, channel: str) -> dict[str, object] | None:
        key = channel if channel in ("local", "remote") else "local"
        task = self._current_task_by_channel.get(key)
        if task is None:
            return None
        return {
            "channel": task.channel,
            "text": task.text,
            "utterance_id": task.utterance_id,
            "revision": task.revision,
            "created_at": task.created_at,
            "is_final": task.is_final,
            "is_stable_partial": task.is_stable_partial,
            "is_early_final": task.is_early_final,
        }

    def _run_passthrough(self, channel: str) -> None:
        while not self._passthrough_stop.is_set():
            task = self._next_passthrough_task(channel)
            if task is None:
                continue
            if self._passthrough_stop.is_set():
                break
            if self._output_mode.get(channel) != "passthrough":
                continue
            try:
                self._playbacks[channel].play(
                    audio=task.audio,
                    sample_rate=int(round(task.sample_rate)),
                    output_device_name=self._output_getters[channel](),
                )
            except Exception as exc:
                if self._on_error:
                    self._on_error(
                        ErrorEvent(
                            level="warning",
                            module="tts_manager",
                            source=channel,
                            code="passthrough_playback_failed",
                            message="Passthrough playback failed",
                            detail=str(exc),
                        )
                    )

    def _next_passthrough_task(self, channel: str) -> _PassthroughTask | None:
        with self._queue_changed:
            while not self._passthrough_stop.is_set() and not self._passthrough_pending[channel]:
                self._queue_changed.wait(timeout=0.2)
            if self._passthrough_stop.is_set():
                return None
            queue = self._passthrough_pending[channel]
            if not queue:
                return None
            first = queue.popleft()

            # Merge adjacent chunks into one packet (up to ~320ms) so playback has fewer gaps.
            merged_audio = [first.audio]
            total_frames = int(first.audio.shape[0])
            max_frames = max(1, int(float(first.sample_rate) * 0.32))
            merged_chunks = 1
            while queue and total_frames < max_frames and merged_chunks < 8:
                nxt = queue[0]
                if abs(float(nxt.sample_rate) - float(first.sample_rate)) > 1.0:
                    break
                queue.popleft()
                merged_audio.append(nxt.audio)
                total_frames += int(nxt.audio.shape[0])
                merged_chunks += 1

            if len(merged_audio) == 1:
                return _PassthroughTask(
                    audio=self._apply_edge_fade(first.audio, first.sample_rate),
                    sample_rate=first.sample_rate,
                    created_at=first.created_at,
                )
            return _PassthroughTask(
                audio=self._apply_edge_fade(np.concatenate(merged_audio, axis=0), first.sample_rate),
                sample_rate=float(first.sample_rate),
                created_at=first.created_at,
            )

    @staticmethod
    def _apply_edge_fade(audio: np.ndarray, sample_rate: float) -> np.ndarray:
        # Short fade-in/out removes start/end clicks when passthrough packets are played back chunk-by-chunk.
        if audio.size == 0 or sample_rate <= 0:
            return audio
        frames = int(audio.shape[0]) if audio.ndim > 1 else int(audio.shape[0])
        if frames <= 8:
            return audio
        fade_frames = min(max(1, int(sample_rate * 0.008)), frames // 4)
        if fade_frames <= 0:
            return audio
        ramp = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
        out = audio.astype(np.float32, copy=True)
        if out.ndim == 1:
            out[:fade_frames] *= ramp
            out[-fade_frames:] *= ramp[::-1]
            return out
        out[:fade_frames, :] *= ramp.reshape(-1, 1)
        out[-fade_frames:, :] *= ramp[::-1].reshape(-1, 1)
        return out

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
        # 通道映射：remote 對應遠端翻譯目標(Meeting)，local 對應本地翻譯目標(Local)
        if channel == "remote":
            target_language = self._config.language.local_target
            preferred_voice = str(getattr(self._config.runtime, "local_tts_voice", "none") or "none")
        else:
            target_language = self._config.language.meeting_target
            preferred_voice = str(getattr(self._config.runtime, "remote_tts_voice", "none") or "none")

        if preferred_voice.lower() == "none":
            # allow no voice path to be handled by output mode subtitle_only / passthrough
            return resolve_tts_config_for_target(self._config, target_language)

        resolved_cfg = resolve_tts_config_for_target(self._config, target_language)
        return TtsConfig(
            engine=resolved_cfg.engine,
            executable_path=resolved_cfg.executable_path,
            model_path=resolved_cfg.model_path,
            config_path=resolved_cfg.config_path,
            voice_name=preferred_voice,
            style_preset=resolved_cfg.style_preset,
            speaker_id=resolved_cfg.speaker_id,
            length_scale=resolved_cfg.length_scale,
            noise_scale=resolved_cfg.noise_scale,
            noise_w=resolved_cfg.noise_w,
            sample_rate=resolved_cfg.sample_rate,
        )

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
