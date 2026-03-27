from __future__ import annotations

from dataclasses import dataclass
import time
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
        self._echo_guard_resume_until = {"local": 0.0, "remote": 0.0}

    def start_session(self) -> None:
        with self._lock:
            self._running = True
            self._asr_enabled["local"] = True
            self._asr_enabled["remote"] = True
            self._tts_busy["local"] = False
            self._tts_busy["remote"] = False

    def stop_session(self) -> None:
        with self._lock:
            self._running = False
            self._asr_enabled["local"] = False
            self._asr_enabled["remote"] = False
            self._tts_busy["local"] = False
            self._tts_busy["remote"] = False

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
            if key == "local" and self._local_echo_guard_enabled:
                self._asr_enabled["local"] = False

    def on_tts_end(self, channel: str, resume_delay_ms: int | None = None) -> None:
        key = channel if channel in ("local", "remote") else "local"
        with self._lock:
            self._tts_busy[key] = False
            delay_ms = resume_delay_ms if resume_delay_ms is not None else (
                self._local_resume_delay_ms if key == "local" else self._remote_resume_delay_ms
            )
            self._echo_guard_resume_until[key] = time.time() + max(0, int(delay_ms)) / 1000.0

    def tick(self) -> None:
        now = time.time()
        with self._lock:
            if self._local_echo_guard_enabled and not self._tts_busy["local"]:
                if now >= self._echo_guard_resume_until["local"]:
                    self._asr_enabled["local"] = True
            # remote echo guard supports only delay sync (no enable flag in runtime currently)
            if not self._tts_busy["remote"] and now >= self._echo_guard_resume_until["remote"]:
                self._asr_enabled["remote"] = True

    def snapshot(self) -> StateSnapshot:
        self.tick()
        with self._lock:
            return StateSnapshot(
                running=self._running,
                local_asr_enabled=self._asr_enabled["local"],
                remote_asr_enabled=self._asr_enabled["remote"],
                local_tts_busy=self._tts_busy["local"],
                remote_tts_busy=self._tts_busy["remote"],
                local_resume_in_ms=0,
                remote_resume_in_ms=0,
            )


RuntimeStateManager = StateManager

__all__ = ["StateSnapshot", "StateManager", "RuntimeStateManager"]
