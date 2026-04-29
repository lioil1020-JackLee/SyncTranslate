"""Shim -- LoudnessStage has moved to stages.py."""
from app.infra.asr.audio_pipeline.stages import LoudnessStage as LoudnessStage  # noqa: F401

__all__ = ["LoudnessStage"]
