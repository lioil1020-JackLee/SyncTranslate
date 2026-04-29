"""Static config read/write helpers for LiveCaptionPage.

Extracted to keep live_caption_page.py focused on UI state and event handling.
Import _LiveCaptionConfigMixin and _CHANNEL_DEFAULTS from here.
"""
from __future__ import annotations

from pathlib import Path

from app.infra.config.schema import AppConfig

_CHANNEL_DEFAULTS: dict[str, dict[str, str]] = {
    "remote": {
        "asr_default": "zh-TW",
        "target_default": "zh-TW",
        "source_default": "en",
        "original_label": "遠端原文",
        "translated_label": "遠端翻譯",
        "output_label": "遠端輸出",
    },
    "local": {
        "asr_default": "en",
        "target_default": "en",
        "source_default": "zh-TW",
        "original_label": "本地原文",
        "translated_label": "本地翻譯",
        "output_label": "本地輸出",
    },
}


class _LiveCaptionConfigMixin:
    """Pure config read/write statics with no UI state dependency."""

    @staticmethod
    def _config_source_language(config: AppConfig, channel: str) -> str:
        if channel == "remote":
            return str(getattr(config.language, "meeting_source", _CHANNEL_DEFAULTS[channel]["source_default"]) or _CHANNEL_DEFAULTS[channel]["source_default"])
        return str(getattr(config.language, "local_source", _CHANNEL_DEFAULTS[channel]["source_default"]) or _CHANNEL_DEFAULTS[channel]["source_default"])

    @staticmethod
    def _config_asr_language(config: AppConfig, channel: str) -> str:
        attr = "remote_asr_language" if channel == "remote" else "local_asr_language"
        return str(getattr(config.runtime, attr, "") or "")

    @staticmethod
    def _asr_model_label_from_profile(asr_cfg) -> str:
        raw = str(getattr(asr_cfg, "model", "") or "large-v3-turbo").strip() or "large-v3-turbo"
        normalized = raw.replace("/", "\\").rstrip("\\")
        if "\\" in normalized:
            return Path(normalized).name or raw
        return raw

    @staticmethod
    def _config_translation_target(config: AppConfig, channel: str) -> str:
        runtime_attr = "remote_translation_target" if channel == "remote" else "local_translation_target"
        language_attr = "meeting_target" if channel == "remote" else "local_target"
        fallback = _CHANNEL_DEFAULTS[channel]["target_default"]
        return str(getattr(config.runtime, runtime_attr, getattr(config.language, language_attr, fallback) or fallback) or "none")

    @staticmethod
    def _config_tts_voice(config: AppConfig, channel: str) -> str:
        attr = "remote_tts_voice" if channel == "remote" else "local_tts_voice"
        return str(getattr(config.runtime, attr, "none") or "none")

    @staticmethod
    def _config_tts_enabled(config: AppConfig, channel: str) -> bool:
        attr = "remote_tts_enabled" if channel == "remote" else "local_tts_enabled"
        return bool(getattr(config.runtime, attr, False))

    @staticmethod
    def _set_runtime_translation_target(config: AppConfig, channel: str, value: str) -> None:
        setattr(config.runtime, f"{channel}_translation_target", value)

    @staticmethod
    def _set_runtime_translation_enabled(config: AppConfig, channel: str, enabled: bool) -> None:
        setattr(config.runtime, f"{channel}_translation_enabled", enabled)

    @staticmethod
    def _set_runtime_asr_language(config: AppConfig, channel: str, value: str) -> None:
        setattr(config.runtime, f"{channel}_asr_language", value)

    @staticmethod
    def _set_runtime_tts_voice(config: AppConfig, channel: str, value: str) -> None:
        setattr(config.runtime, f"{channel}_tts_voice", value)

    @staticmethod
    def _set_runtime_tts_enabled(config: AppConfig, channel: str, enabled: bool) -> None:
        setattr(config.runtime, f"{channel}_tts_enabled", enabled)

    @staticmethod
    def _set_language_target(config: AppConfig, channel: str, value: str) -> None:
        attr = "meeting_target" if channel == "remote" else "local_target"
        setattr(config.language, attr, value)
