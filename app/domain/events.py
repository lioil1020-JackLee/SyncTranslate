"""Shim -- PipelineMeta and ErrorEvent have moved to app.domain.models."""
from app.domain.models import ErrorEvent as ErrorEvent  # noqa: F401
from app.domain.models import PipelineMeta as PipelineMeta  # noqa: F401

__all__ = ["ErrorEvent", "PipelineMeta"]
