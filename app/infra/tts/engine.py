from __future__ import annotations

# Shim — functions moved to app.infra.tts.playback_queue
from app.infra.tts.playback_queue import (  # noqa: F401
    create_tts_engine,
    edge_tts_rate_for_style,
)

__all__ = ["create_tts_engine", "edge_tts_rate_for_style"]
