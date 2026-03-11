from __future__ import annotations

from dataclasses import dataclass

from app.pipeline_direction import DirectionalPipeline
from app.schemas import AudioRouteConfig


@dataclass(slots=True)
class SessionResult:
    ok: bool
    message: str


class SessionController:
    def __init__(self, meeting_pipeline: DirectionalPipeline, local_pipeline: DirectionalPipeline) -> None:
        self._meeting_pipeline = meeting_pipeline
        self._local_pipeline = local_pipeline

    def is_running(self) -> bool:
        return self._meeting_pipeline.running or self._local_pipeline.running

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int, chunk_ms: int = 100) -> SessionResult:
        try:
            if mode == "meeting_to_local":
                self._meeting_pipeline.start(routes.meeting_in, sample_rate, chunk_ms=chunk_ms)
            elif mode == "local_to_meeting":
                self._local_pipeline.start(routes.microphone_in, sample_rate, chunk_ms=chunk_ms)
            elif mode == "bidirectional":
                self._meeting_pipeline.start(routes.meeting_in, sample_rate, chunk_ms=chunk_ms)
                self._local_pipeline.start(routes.microphone_in, sample_rate, chunk_ms=chunk_ms)
            else:
                return SessionResult(ok=False, message=f"未知模式: {mode}")
        except Exception as exc:
            self.stop()
            return SessionResult(ok=False, message=str(exc))
        return SessionResult(ok=True, message=f"Session started: {mode}")

    def stop(self) -> SessionResult:
        self._meeting_pipeline.stop()
        self._local_pipeline.stop()
        return SessionResult(ok=True, message="Session stopped")
