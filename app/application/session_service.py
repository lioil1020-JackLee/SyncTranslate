from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from threading import Lock
import time
from typing import TYPE_CHECKING, Any

from app.application.audio_router import AudioRouter
from app.domain.constants import ASR_DEFAULT_CHUNK_MS
from app.infra.config.schema import AppConfig, AudioRouteConfig

if TYPE_CHECKING:
    from app.bootstrap.dependency_container import PipelineBundle


@dataclass(slots=True)
class SessionResult:
    ok: bool
    message: str
    payload: dict[str, Any] | None = None


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class SessionController:
    def __init__(self, audio_router: AudioRouter) -> None:
        self._audio_router = audio_router
        self._state = SessionState.IDLE
        self._lock = Lock()
        self._last_failure: str = ""
        self._started_at: float | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._state == SessionState.RUNNING

    def current_state(self) -> SessionState:
        with self._lock:
            return self._state

    def last_failure(self) -> str:
        with self._lock:
            return self._last_failure

    def start(
        self,
        routes: AudioRouteConfig,
        sample_rate: int,
        chunk_ms: int = ASR_DEFAULT_CHUNK_MS,
        *,
        mode: str = "bidirectional",
    ) -> SessionResult:
        with self._lock:
            if self._state in (SessionState.STARTING, SessionState.STOPPING):
                return SessionResult(ok=False, message=f"Session is {self._state.value}, try again shortly")
            if self._state == SessionState.RUNNING:
                return SessionResult(ok=True, message="Session already running", payload={"mode": mode})
            self._state = SessionState.STARTING
            self._last_failure = ""

        try:
            self._audio_router.start(mode, routes, sample_rate, chunk_ms=chunk_ms)
        except Exception as exc:
            try:
                self._audio_router.stop()
            except Exception:
                pass
            with self._lock:
                self._state = SessionState.IDLE
                self._last_failure = str(exc)
            return SessionResult(ok=False, message=str(exc))
        with self._lock:
            self._state = SessionState.RUNNING
            self._started_at = time.time()
        return SessionResult(
            ok=True,
            message=f"Session started: {mode}",
            payload={"mode": mode, "state": SessionState.RUNNING.value},
        )

    def stop(self) -> SessionResult:
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.FAILED):
                try:
                    self._audio_router.stop()
                except Exception:
                    pass
                self._state = SessionState.IDLE
                self._started_at = None
                return SessionResult(ok=True, message="Session already stopped", payload={"stats_before_stop": {}})
            if self._state == SessionState.STOPPING:
                return SessionResult(ok=False, message="Session is stopping")
            self._state = SessionState.STOPPING

        stats_before_stop = asdict(self._audio_router.stats())
        try:
            self._audio_router.stop()
        except Exception as exc:
            with self._lock:
                self._state = SessionState.FAILED
                self._last_failure = str(exc)
            return SessionResult(ok=False, message=f"Session stop failed: {exc}")

        ended_at = time.time()
        with self._lock:
            started_at = self._started_at
            self._state = SessionState.IDLE
            self._started_at = None
        duration_sec = max(0.0, ended_at - started_at) if started_at else 0.0
        return SessionResult(
            ok=True,
            message="Session stopped",
            payload={
                "stats_before_stop": stats_before_stop,
                "session_meta": {
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_sec": duration_sec,
                },
            },
        )

    def reset_failed(self) -> SessionResult:
        """將 FAILED 狀態重設為 IDLE，允許重試啟動。"""
        with self._lock:
            if self._state != SessionState.FAILED:
                return SessionResult(ok=False, message=f"Session is not in FAILED state (current: {self._state.value})")
            self._state = SessionState.IDLE
            self._last_failure = ""
        return SessionResult(ok=True, message="Session reset to IDLE")


SessionService = SessionController


class RuntimeFacade:
    def __init__(self, builder: Callable[[AppConfig, int], PipelineBundle]) -> None:
        self._builder = builder
        self._bundle: PipelineBundle | None = None
        self._dirty = True
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def bundle(self) -> PipelineBundle | None:
        return self._bundle

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild(self, config: AppConfig) -> PipelineBundle:
        self._revision += 1
        self._bundle = self._builder(config, self._revision)
        self._dirty = False
        return self._bundle

    def ensure_ready(self, config: AppConfig) -> PipelineBundle:
        if self._bundle is None or self._dirty:
            return self.rebuild(config)
        return self._bundle


RuntimeOrchestrator = RuntimeFacade


__all__ = [
    "RuntimeFacade",
    "RuntimeOrchestrator",
    "SessionController",
    "SessionResult",
    "SessionService",
    "SessionState",
]
