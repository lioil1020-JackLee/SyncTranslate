from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from app.infra.asr.language_profiles import normalize_asr_language
from app.infra.config.schema import AppConfig, AsrConfig


def is_chinese_language(language: str) -> bool:
    return normalize_asr_language(language) == "zh"


def alternate_chinese_model(current_model: str, configured_fallback: str = "") -> str:
    current = str(current_model or "").strip()
    fallback = str(configured_fallback or "").strip()
    if fallback and Path(fallback).as_posix().lower() != Path(current).as_posix().lower():
        return fallback
    lowered = current.replace("\\", "/").lower()
    if "belle" in lowered:
        return "large-v3-turbo"
    return r".\runtimes\models\belle-zh-ct2"


def build_chinese_fallback_profile(config: AppConfig, base_profile: AsrConfig, *, language: str) -> AsrConfig | None:
    runtime = config.runtime
    if not bool(getattr(runtime, "asr_chinese_fallback_enabled", True)):
        return None
    if str(getattr(runtime, "asr_accuracy_mode", "balanced") or "balanced") == "low_latency":
        return None
    if not is_chinese_language(language):
        return None

    fallback_model = alternate_chinese_model(
        str(getattr(base_profile, "model", "") or ""),
        str(getattr(runtime, "asr_chinese_fallback_model", "") or ""),
    )
    if not fallback_model or fallback_model == str(getattr(base_profile, "model", "") or ""):
        return None

    profile = deepcopy(base_profile)
    profile.model = fallback_model
    profile.beam_size = max(1, int(getattr(profile, "beam_size", 1)))
    profile.final_beam_size = max(5, int(getattr(profile, "final_beam_size", 4)))
    profile.temperature_fallback = "0.0"
    return profile


__all__ = [
    "alternate_chinese_model",
    "build_chinese_fallback_profile",
    "is_chinese_language",
]
