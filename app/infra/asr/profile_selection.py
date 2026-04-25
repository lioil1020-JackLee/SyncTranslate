from __future__ import annotations

from collections.abc import Iterable, Iterator

from app.infra.asr.backend_resolution import resolve_backend_for_language
from app.infra.config.schema import AppConfig, AsrConfig


CHINESE_ASR_PROFILE_SLOT = "local"
NON_CHINESE_ASR_PROFILE_SLOT = "remote"
DISABLED_ASR_PROFILE_SLOT = "disabled"


def requested_asr_language_for_source(config: AppConfig, source: str) -> str:
    runtime = config.runtime
    explicit = (
        str(getattr(runtime, "remote_asr_language", "") or "").strip()
        if source == "remote"
        else str(getattr(runtime, "local_asr_language", "") or "").strip()
    )
    if explicit:
        if explicit.lower() == "auto":
            return ""
        return explicit

    language_mode = str(getattr(runtime, "asr_language_mode", "") or "").strip().lower()
    if language_mode == "auto":
        return ""
    if source == "remote":
        return str(getattr(config.language, "meeting_source", "") or "").strip()
    return str(getattr(config.language, "local_source", "") or "").strip()


def asr_profile_slot_for_language(language: str) -> str:
    resolution = resolve_backend_for_language(language)
    if resolution.disabled:
        return DISABLED_ASR_PROFILE_SLOT
    if resolution.language_family == "chinese":
        return CHINESE_ASR_PROFILE_SLOT
    return NON_CHINESE_ASR_PROFILE_SLOT


def asr_profile_family_for_language(language: str) -> str:
    slot = asr_profile_slot_for_language(language)
    if slot == CHINESE_ASR_PROFILE_SLOT:
        return "chinese"
    if slot == DISABLED_ASR_PROFILE_SLOT:
        return "disabled"
    return "non_chinese"


def asr_profile_for_language(config: AppConfig, language: str) -> AsrConfig:
    slot = asr_profile_slot_for_language(language)
    if slot == CHINESE_ASR_PROFILE_SLOT:
        return config.asr_channels.local
    return config.asr_channels.remote


def asr_profile_for_source(config: AppConfig, source: str) -> AsrConfig:
    return asr_profile_for_language(config, requested_asr_language_for_source(config, source))


def iter_active_asr_profiles_for_sources(
    config: AppConfig,
    sources: Iterable[str] = ("remote", "local"),
) -> Iterator[tuple[str, str, AsrConfig]]:
    seen: set[tuple[str, str, str, str]] = set()
    for source in sources:
        language = requested_asr_language_for_source(config, source)
        slot = asr_profile_slot_for_language(language)
        if slot == DISABLED_ASR_PROFILE_SLOT:
            continue
        profile = asr_profile_for_language(config, language)
        key = (
            str(getattr(profile, "engine", "") or ""),
            str(getattr(profile, "model", "") or ""),
            str(getattr(profile, "device", "") or ""),
            str(getattr(profile, "compute_type", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        yield source, language, profile

