from __future__ import annotations

from dataclasses import replace

from app.infra.config.schema import AppConfig, TtsConfig, merge_tts_configs


def normalize_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if "_" in normalized:
        normalized = normalized.replace("_", "-")
    return normalized


def voice_matches_language(voice_name: str, language: str) -> bool:
    lowered = (voice_name or "").strip().lower()
    normalized_language = normalize_language(language)
    if not lowered:
        return False
    if normalized_language.startswith("zh"):
        return lowered.startswith("zh-")
    if normalized_language.startswith("en"):
        return lowered.startswith("en-")
    if normalized_language.startswith("ja"):
        return lowered.startswith("ja-")
    if normalized_language.startswith("ko"):
        return lowered.startswith("ko-")
    if normalized_language.startswith("th"):
        return lowered.startswith("th-")
    return True


def default_voice_for_language(language: str) -> str:
    normalized_language = normalize_language(language)
    if normalized_language.startswith("zh"):
        return "zh-TW-HsiaoChenNeural"
    if normalized_language.startswith("en"):
        return "en-US-JennyNeural"
    if normalized_language.startswith("ja"):
        return "ja-JP-NanamiNeural"
    if normalized_language.startswith("ko"):
        return "ko-KR-SunHiNeural"
    if normalized_language.startswith("th"):
        return "th-TH-PremwadeeNeural"
    return ""


def resolve_tts_config_for_target(config: AppConfig, target_language: str) -> TtsConfig:
    normalized = normalize_language(target_language)
    base, override = _select_language_profile(config, normalized)
    resolved = merge_tts_configs(config.tts, base, override)
    if (resolved.engine or "").strip().lower() != "edge_tts":
        return resolved
    if voice_matches_language(resolved.voice_name, normalized):
        return resolved
    fallback_voice = default_voice_for_language(normalized)
    if not fallback_voice:
        return resolved
    return replace(resolved, voice_name=fallback_voice)


def resolve_edge_voice_for_target(config: AppConfig, target_language: str) -> str:
    resolved = resolve_tts_config_for_target(config, target_language)
    voice = (resolved.voice_name or "").strip()
    if voice:
        return voice
    fallback = default_voice_for_language(target_language)
    if fallback:
        return fallback
    return (resolved.model_path or "").strip()


def _select_language_profile(config: AppConfig, normalized_language: str):
    if normalized_language.startswith("en"):
        return config.local_tts, config.tts_channels.remote
    if normalized_language.startswith("zh"):
        return config.meeting_tts, config.tts_channels.local
    if voice_matches_language(config.local_tts.voice_name, normalized_language):
        return config.local_tts, config.tts_channels.remote
    return config.meeting_tts, config.tts_channels.local
