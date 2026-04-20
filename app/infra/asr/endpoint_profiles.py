"""Endpoint profiles — predefined ASR endpointing configurations.

Profiles control VAD sensitivity, timing thresholds, and partial-emission
behaviour for different acoustic environments and use-cases.

Usage
-----
from app.infra.asr.endpoint_profiles import get_endpoint_profile, PROFILES

profile = get_endpoint_profile("meeting_room")
# Apply fields to SourceRuntimeV2 constructor kwargs
kwargs.update(profile.to_worker_kwargs())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EndpointProfile:
    """Collection of endpointing tunable parameters for SourceRuntimeV2."""

    name: str

    # --- partial emission ---
    partial_interval_ms: int = 500
    min_partial_audio_ms: int = 300

    # --- finalization thresholds ---
    soft_final_audio_ms: int = 3600
    soft_endpoint_finalize_audio_ms: int = 1200
    speech_end_finalize_audio_ms: int = 1000
    pre_roll_ms: int = 200

    # --- VAD backend preference (informational, not enforced here) ---
    preferred_vad_backend: str = "auto"

    # --- description ---
    description: str = ""

    def to_worker_kwargs(self) -> dict[str, Any]:
        """Return only the kwargs accepted by SourceRuntimeV2.__init__."""
        return {
            "partial_interval_ms": self.partial_interval_ms,
            "min_partial_audio_ms": self.min_partial_audio_ms,
            "soft_final_audio_ms": self.soft_final_audio_ms,
            "pre_roll_ms": self.pre_roll_ms,
        }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, EndpointProfile] = {
    "default": EndpointProfile(
        name="default",
        partial_interval_ms=500,
        min_partial_audio_ms=300,
        soft_final_audio_ms=3600,
        soft_endpoint_finalize_audio_ms=1200,
        speech_end_finalize_audio_ms=1000,
        pre_roll_ms=200,
        preferred_vad_backend="auto",
        description="Balanced profile for general desktop use.",
    ),
    "meeting_room": EndpointProfile(
        name="meeting_room",
        partial_interval_ms=600,
        min_partial_audio_ms=400,
        soft_final_audio_ms=4000,
        soft_endpoint_finalize_audio_ms=1400,
        speech_end_finalize_audio_ms=1200,
        pre_roll_ms=250,
        preferred_vad_backend="silero",
        description="Slightly longer pauses for multi-speaker meeting room audio.",
    ),
    "headset": EndpointProfile(
        name="headset",
        partial_interval_ms=500,
        min_partial_audio_ms=260,
        soft_final_audio_ms=4000,
        soft_endpoint_finalize_audio_ms=1200,
        speech_end_finalize_audio_ms=1000,
        pre_roll_ms=180,
        preferred_vad_backend="silero",
        description="Balanced headset profile tuned to reduce over-segmentation.",
    ),
    "noisy_environment": EndpointProfile(
        name="noisy_environment",
        partial_interval_ms=700,
        min_partial_audio_ms=500,
        soft_final_audio_ms=4800,
        soft_endpoint_finalize_audio_ms=1800,
        speech_end_finalize_audio_ms=1500,
        pre_roll_ms=300,
        preferred_vad_backend="silero_vad",
        description="Conservative finalization in high-noise environments.",
    ),
    "max_accuracy": EndpointProfile(
        name="max_accuracy",
        partial_interval_ms=800,
        min_partial_audio_ms=600,
        soft_final_audio_ms=6000,
        soft_endpoint_finalize_audio_ms=2200,
        speech_end_finalize_audio_ms=2000,
        pre_roll_ms=400,
        preferred_vad_backend="silero_vad",
        description="Maximize transcription accuracy at the cost of latency.",
    ),
    "low_latency": EndpointProfile(
        name="low_latency",
        partial_interval_ms=200,
        min_partial_audio_ms=150,
        soft_final_audio_ms=2000,
        soft_endpoint_finalize_audio_ms=600,
        speech_end_finalize_audio_ms=500,
        pre_roll_ms=100,
        preferred_vad_backend="silero",
        description="Minimize partial and final latency; may reduce accuracy.",
    ),
    "turn_taking": EndpointProfile(
        name="turn_taking",
        partial_interval_ms=200,
        min_partial_audio_ms=180,
        soft_final_audio_ms=1600,
        soft_endpoint_finalize_audio_ms=520,
        speech_end_finalize_audio_ms=420,
        pre_roll_ms=80,
        preferred_vad_backend="silero",
        description="Fast hand-off profile for short, back-and-forth conversation turns.",
    ),
}


def get_endpoint_profile(name: str | None) -> EndpointProfile:
    """Return the named profile, falling back to 'default' if not found."""
    if not name:
        return PROFILES["default"]
    return PROFILES.get(name, PROFILES["default"])


def list_profiles() -> list[str]:
    """Return sorted list of built-in profile names."""
    return sorted(PROFILES.keys())


__all__ = [
    "EndpointProfile",
    "PROFILES",
    "get_endpoint_profile",
    "list_profiles",
]
