"""Shim -- IdentityStage has moved to stages.py."""
from app.infra.asr.audio_pipeline.stages import IdentityStage as IdentityStage  # noqa: F401

__all__ = ["IdentityStage"]
