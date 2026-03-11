from __future__ import annotations

from dataclasses import dataclass

from app.pipeline_local import LocalPipeline
from app.pipeline_remote import RemotePipeline
from app.schemas import AudioRouteConfig


@dataclass(slots=True)
class SessionResult:
    ok: bool
    message: str


class SessionController:
    def __init__(self, remote_pipeline: RemotePipeline, local_pipeline: LocalPipeline) -> None:
        self._remote_pipeline = remote_pipeline
        self._local_pipeline = local_pipeline

    def is_running(self) -> bool:
        return self._remote_pipeline.running or self._local_pipeline.running

    def start(self, mode: str, routes: AudioRouteConfig, sample_rate: int) -> SessionResult:
        try:
            if mode == "remote_only":
                self._remote_pipeline.start(routes.remote_in, sample_rate)
            elif mode == "local_only":
                self._local_pipeline.start(routes.local_mic_in, sample_rate)
            elif mode == "bidirectional":
                self._remote_pipeline.start(routes.remote_in, sample_rate)
                self._local_pipeline.start(routes.local_mic_in, sample_rate)
            else:
                return SessionResult(ok=False, message=f"未知模式: {mode}")
        except Exception as exc:
            # If one side failed during startup, stop all to avoid half-broken session.
            self.stop()
            return SessionResult(ok=False, message=str(exc))
        return SessionResult(ok=True, message=f"Session started: {mode}")

    def stop(self) -> SessionResult:
        self._remote_pipeline.stop()
        self._local_pipeline.stop()
        return SessionResult(ok=True, message="Session stopped")

