"""Shim -- TranscriptItem has moved to app.domain.models."""
from app.domain.models import TranscriptItem as TranscriptItem  # noqa: F401

__all__ = ["TranscriptItem"]
