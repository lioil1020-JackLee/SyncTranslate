"""Shim -- HighpassStage has moved to stages.py."""
from app.infra.asr.audio_pipeline.stages import HighpassStage as HighpassStage  # noqa: F401

__all__ = ["HighpassStage"]
