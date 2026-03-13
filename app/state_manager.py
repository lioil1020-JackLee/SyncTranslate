from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class StateSnapshot:
    running: bool
    local_asr_enabled: bool
    remote_asr_enabled: bool
    local_tts_busy: bool
    remote_tts_busy: bool
    local_resume_in_ms: int
    remote_resume_in_ms: int


class StateManager:
    def __init__(
        self,
        *,
        local_echo_guard_enabled: bool = False,
        local_resume_delay_ms: int = 300,
        remote_resume_delay_ms: int = 300,
    ) -> None:
        self._lock = Lock()
        self._running = False
        self._asr_enabled = {"local": True, "remote": True}
        self._tts_busy = {"local": False, "remote": False}
        self._local_echo_guard_enabled = bool(local_echo_guard_enabled)
        self._local_resume_delay_ms = max(0, int(local_resume_delay_ms))
        self._remote_resume_delay_ms = max(0, int(remote_resume_delay_ms))
        self._local_resume_at = 0.0
        self._remote_resume_at = 0.0

    def start_session(self) -> None:
        with self._lock:
            self._running = True
            self._asr_enabled["local"] = True
            self._asr_enabled["remote"] = True
            self._tts_busy["local"] = False
            self._tts_busy["remote"] = False
            self._local_resume_at = 0.0
            self._remote_resume_at = 0.0

    def stop_session(self) -> None:
        with self._lock:
            self._running = False
            self._asr_enabled["local"] = False
            self._asr_enabled["remote"] = False
            self._tts_busy["local"] = False
            self._tts_busy["remote"] = False
            self._local_resume_at = 0.0
            self._remote_resume_at = 0.0

    def set_asr_enabled(self, source: str, enabled: bool) -> None:
        key = source if source in ("local", "remote") else "local"
        with self._lock:
            self._asr_enabled[key] = bool(enabled)

    def can_accept_asr(self, source: str) -> bool:
        self.tick()
        key = source if source in ("local", "remote") else "local"
        with self._lock:
            return self._running and self._asr_enabled.get(key, False)

    def on_tts_start(self, channel: str) -> None:
        key = channel if channel in ("local", "remote") else "local"
        with self._lock:
            self._tts_busy[key] = True
            if key == "remote":
                self._asr_enabled["remote"] = False
                self._remote_resume_at = 0.0
            elif key == "local" and self._local_echo_guard_enabled:
                self._asr_enabled["local"] = False
                self._local_resume_at = 0.0

    def on_tts_end(self, channel: str, resume_delay_ms: int | None = None) -> None:
        key = channel if channel in ("local", "remote") else "local"
        if resume_delay_ms is None:
            delay_ms = self._remote_resume_delay_ms if key == "remote" else self._local_resume_delay_ms
        else:
            delay_ms = max(0, int(resume_delay_ms))
        with self._lock:
            self._tts_busy[key] = False
            if key == "remote":
                self._remote_resume_at = time.monotonic() + (delay_ms / 1000.0)
            elif key == "local" and self._local_echo_guard_enabled:
                self._local_resume_at = time.monotonic() + (delay_ms / 1000.0)

    def tick(self) -> None:
        with self._lock:
            if not self._running:
                return
            if self._remote_resume_at <= 0.0:
                pass
            elif time.monotonic() >= self._remote_resume_at:
                self._asr_enabled["remote"] = True
                self._remote_resume_at = 0.0
            if self._local_resume_at <= 0.0:
                return
            if time.monotonic() < self._local_resume_at:
                return
            self._asr_enabled["local"] = True
            self._local_resume_at = 0.0

    def snapshot(self) -> StateSnapshot:
        self.tick()
        with self._lock:
            if self._local_resume_at <= 0.0:
                local_resume_in_ms = 0
            else:
                local_resume_in_ms = max(0, int(round((self._local_resume_at - time.monotonic()) * 1000.0)))
            if self._remote_resume_at <= 0.0:
                resume_in_ms = 0
            else:
                resume_in_ms = max(0, int(round((self._remote_resume_at - time.monotonic()) * 1000.0)))
            return StateSnapshot(
                running=self._running,
                local_asr_enabled=self._asr_enabled["local"],
                remote_asr_enabled=self._asr_enabled["remote"],
                local_tts_busy=self._tts_busy["local"],
                remote_tts_busy=self._tts_busy["remote"],
                local_resume_in_ms=local_resume_in_ms,
                remote_resume_in_ms=resume_in_ms,
            )
