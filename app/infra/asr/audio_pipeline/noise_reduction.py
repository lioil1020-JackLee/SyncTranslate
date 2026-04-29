"""Shim -- NoiseReductionStage has moved to stages.py."""
from app.infra.asr.audio_pipeline.stages import NoiseReductionStage as NoiseReductionStage  # noqa: F401

__all__ = ["NoiseReductionStage"]
