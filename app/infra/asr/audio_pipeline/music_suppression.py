"""Shim -- MusicSuppressionStage has moved to stages.py."""
from app.infra.asr.audio_pipeline.stages import MusicSuppressionStage as MusicSuppressionStage  # noqa: F401

__all__ = ["MusicSuppressionStage"]
