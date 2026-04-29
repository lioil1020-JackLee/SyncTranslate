"""Config key normalization/presentation helpers — extracted from settings_store."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.infra.config._config_migration import _normalize_asr_profile_legacy_fields


def _normalize_external_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(raw)
    asr_profiles = data.get("asr_profiles")
    if isinstance(asr_profiles, dict):
        asr_channels = data.get("asr_channels")
        if not isinstance(asr_channels, dict):
            asr_channels = {}
        if "chinese" in asr_profiles and "local" not in asr_channels:
            asr_channels["local"] = asr_profiles.get("chinese")
        if "non_chinese" in asr_profiles and "remote" not in asr_channels:
            asr_channels["remote"] = asr_profiles.get("non_chinese")
        if "general" in asr_profiles and "remote" not in asr_channels:
            asr_channels["remote"] = asr_profiles.get("general")
        data["asr_channels"] = asr_channels
        data.pop("asr_profiles", None)
    asr = data.get("asr")
    if isinstance(asr, dict):
        _normalize_asr_profile_legacy_fields(asr)
    language = data.get("language")
    if isinstance(language, dict):
        if "remote_translation_target" in language and "meeting_target" not in language:
            language["meeting_target"] = language.get("remote_translation_target")
        if "local_translation_target" in language and "local_target" not in language:
            language["local_target"] = language.get("local_translation_target")
        language.pop("remote_translation_target", None)
        language.pop("local_translation_target", None)
    asr_channels = data.get("asr_channels")
    if isinstance(asr_channels, dict):
        if "chinese" in asr_channels and "local" not in asr_channels:
            asr_channels["local"] = asr_channels.get("chinese")
        if "non_chinese" in asr_channels and "remote" not in asr_channels:
            asr_channels["remote"] = asr_channels.get("non_chinese")
        if "general" in asr_channels and "remote" not in asr_channels:
            asr_channels["remote"] = asr_channels.get("general")
        if "english" in asr_channels and "remote" not in asr_channels:
            asr_channels["remote"] = asr_channels.get("english")
        for key in ("local", "remote"):
            profile = asr_channels.get(key)
            if isinstance(profile, dict):
                _normalize_asr_profile_legacy_fields(profile)
    llm_channels = data.get("llm_channels")
    if isinstance(llm_channels, dict):
        if "zh_to_en" in llm_channels and "local" not in llm_channels:
            llm_channels["local"] = llm_channels.get("zh_to_en")
        if "en_to_zh" in llm_channels and "remote" not in llm_channels:
            llm_channels["remote"] = llm_channels.get("en_to_zh")
    tts_channels = data.get("tts_channels")
    if isinstance(tts_channels, dict):
        if "chinese" in tts_channels and "local" not in tts_channels:
            tts_channels["local"] = tts_channels.get("chinese")
        if "english" in tts_channels and "remote" not in tts_channels:
            tts_channels["remote"] = tts_channels.get("english")
    if "chinese_tts" in data and "meeting_tts" not in data:
        data["meeting_tts"] = data.get("chinese_tts")
    if "english_tts" in data and "local_tts" not in data:
        data["local_tts"] = data.get("english_tts")
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        if str(runtime.get("asr_v2_backend", "")).strip().lower() == "funasr_v2":
            runtime["asr_v2_backend"] = "faster_whisper_v2"
        runtime.pop("streaming_profile_local", None)
        runtime.pop("streaming_profile_remote", None)
        mapping = {
            "asr_queue_maxsize_chinese": "asr_queue_maxsize_local",
            "asr_queue_maxsize_english": "asr_queue_maxsize_remote",
            "llm_queue_maxsize_zh_to_en": "llm_queue_maxsize_local",
            "llm_queue_maxsize_en_to_zh": "llm_queue_maxsize_remote",
            "tts_queue_maxsize_chinese": "tts_queue_maxsize_local",
            "tts_queue_maxsize_english": "tts_queue_maxsize_remote",
        }
        for src, dst in mapping.items():
            if src in runtime and dst not in runtime:
                runtime[dst] = runtime.get(src)
        for src in mapping:
            runtime.pop(src, None)
        if "tts_output_mode" not in runtime:
            remote_tts_enabled = bool(runtime.get("remote_tts_enabled", False))
            local_tts_enabled = bool(runtime.get("local_tts_enabled", False))
            runtime["tts_output_mode"] = "tts" if (remote_tts_enabled or local_tts_enabled) else "subtitle_only"
        language_remote_target = ""
        language_local_target = ""
        if isinstance(language, dict):
            language_remote_target = str(language.get("meeting_target") or "").strip()
            language_local_target = str(language.get("local_target") or "").strip()
        remote_target = str(runtime.get("remote_translation_target") or language_remote_target or "").strip()
        local_target = str(runtime.get("local_translation_target") or language_local_target or "").strip()
        remote_enabled_default = remote_target.lower() != "none" if remote_target else True
        local_enabled_default = local_target.lower() != "none" if local_target else True
        legacy_translation_enabled = bool(runtime.get("translation_enabled", remote_enabled_default or local_enabled_default))
        if "translation_enabled" not in runtime:
            runtime["translation_enabled"] = legacy_translation_enabled
        if "remote_translation_enabled" not in runtime:
            runtime["remote_translation_enabled"] = remote_enabled_default
        if "local_translation_enabled" not in runtime:
            runtime["local_translation_enabled"] = local_enabled_default

        if "remote_asr_language" not in runtime:
            runtime["remote_asr_language"] = "auto"
        if "local_asr_language" not in runtime:
            runtime["local_asr_language"] = "auto"

        if "remote_translation_target" not in runtime:
            enabled = bool(runtime.get("remote_translation_enabled", remote_enabled_default))
            runtime["remote_translation_target"] = remote_target or ("zh-TW" if enabled else "none")
        if "local_translation_target" not in runtime:
            enabled = bool(runtime.get("local_translation_enabled", local_enabled_default))
            runtime["local_translation_target"] = local_target or ("en" if enabled else "none")

        if "remote_tts_voice" not in runtime:
            runtime["remote_tts_voice"] = "none"
        if "local_tts_voice" not in runtime:
            runtime["local_tts_voice"] = "none"

        runtime["asr_language_mode"] = "auto"
    return data


def _present_external_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(raw)
    direction = data.get("direction")
    if isinstance(direction, dict) and str(direction.get("mode") or "bidirectional") == "bidirectional":
        data.pop("direction", None)
    language = data.get("language")
    if isinstance(language, dict):
        language["meeting_source"] = str(language.get("meeting_source") or "en")
        language["meeting_target"] = str(language.get("meeting_target") or language.get("remote_translation_target") or "zh-TW")
        language["local_source"] = str(language.get("local_source") or "zh-TW")
        language["local_target"] = str(language.get("local_target") or language.get("local_translation_target") or "en")
        language.pop("remote_translation_target", None)
        language.pop("local_translation_target", None)
    if "asr_channels" in data:
        data.pop("asr", None)
    if "llm_channels" in data:
        data.pop("llm", None)
    asr_channels = data.get("asr_channels")
    if isinstance(asr_channels, dict):
        asr_profiles: dict[str, Any] = {}
        local_profile = asr_channels.get("local")
        remote_profile = asr_channels.get("remote")
        if isinstance(local_profile, dict):
            asr_profiles["chinese"] = deepcopy(local_profile)
        if isinstance(remote_profile, dict):
            asr_profiles["non_chinese"] = deepcopy(remote_profile)
        data["asr_profiles"] = asr_profiles
        data.pop("asr_channels", None)
    llm_channels = data.get("llm_channels")
    if isinstance(llm_channels, dict):
        llm_channels.pop("zh_to_en", None)
        llm_channels.pop("en_to_zh", None)
    tts_channels = data.get("tts_channels")
    if isinstance(tts_channels, dict):
        tts_channels.pop("chinese", None)
        tts_channels.pop("english", None)
    base_tts = data.get("tts")
    if isinstance(base_tts, dict) and isinstance(tts_channels, dict):
        for channel_key in ("local", "remote"):
            override = tts_channels.get(channel_key)
            if not isinstance(override, dict):
                continue
            cleaned_override: dict[str, Any] = {}
            for key, value in override.items():
                if value is None:
                    continue
                if base_tts.get(key) == value:
                    continue
                cleaned_override[key] = value
            tts_channels[channel_key] = cleaned_override
    data.pop("chinese_tts", None)
    data.pop("english_tts", None)
    data.pop("meeting_tts", None)
    data.pop("local_tts", None)
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        if not bool(runtime.get("remote_translation_enabled", True)):
            runtime["remote_translation_target"] = "none"
        if not bool(runtime.get("local_translation_enabled", True)):
            runtime["local_translation_target"] = "none"
        runtime.pop("asr_queue_maxsize_chinese", None)
        runtime.pop("asr_queue_maxsize_english", None)
        runtime.pop("llm_queue_maxsize_zh_to_en", None)
        runtime.pop("llm_queue_maxsize_en_to_zh", None)
        runtime.pop("tts_queue_maxsize_chinese", None)
        runtime.pop("tts_queue_maxsize_english", None)
        runtime.pop("asr_queue_maxsize", None)
        runtime.pop("llm_queue_maxsize", None)
        runtime.pop("tts_queue_maxsize", None)
        runtime.pop("translation_enabled", None)
        runtime.pop("remote_translation_enabled", None)
        runtime.pop("local_translation_enabled", None)
        runtime.pop("asr_language_mode", None)
        runtime.pop("warmup_on_start", None)
        runtime.pop("use_channel_specific_asr", None)
        runtime.pop("use_channel_specific_llm", None)
        runtime.pop("passthrough_gain", None)
        runtime.pop("tts_gain", None)
        runtime.pop("streaming_profile_local", None)
        runtime.pop("streaming_profile_remote", None)
        runtime.pop("config_schema_version", None)
        runtime.pop("last_migration_note", None)
    health_last_success = data.get("health_last_success")
    if isinstance(health_last_success, dict):
        if not any(str(value or "").strip() for value in health_last_success.values()):
            data.pop("health_last_success", None)
    return data
