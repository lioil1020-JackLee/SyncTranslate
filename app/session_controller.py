from __future__ import annotations

from dataclasses import dataclass

from app.audio_router import AudioRouter
from app.schemas import AudioRouteConfig


@dataclass(slots=True)
class SessionResult:
    ok: bool
    message: str


class SessionController:
    def __init__(self, audio_router: AudioRouter) -> None:
        self._audio_router = audio_router

    def is_running(self) -> bool:
        return self._audio_router.running

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = 100) -> SessionResult:
        try:
            self._audio_router.start(mode, routes, sample_rate, chunk_ms=chunk_ms)
        except Exception as exc:
            self.stop()
            return SessionResult(ok=False, message=str(exc))
        return SessionResult(ok=True, message=f"Session started: {mode}")

    def stop(self) -> SessionResult:
        self._audio_router.stop()
        return SessionResult(ok=True, message="Session stopped")
