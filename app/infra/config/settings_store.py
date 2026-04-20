from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

from app.infra.config.schema import AppConfig



def _normalize_asr_engine_name(value: object) -> str:
    engine = str(value or "").strip().lower()
    if engine == "funasr":
        return "faster_whisper"
    return engine


def _normalize_vad_backend_name(value: object) -> str:
    backend = str(value or "").strip().lower()
    if backend in {"fsmn_vad", "fsmn-vad", "funasr_vad", "funasr"}:
        return "silero_vad"
    if backend in {"", "neural", "neural_endpoint"}:
        return "silero_vad"
    return backend


def _normalize_asr_profile_legacy_fields(profile: dict[str, Any]) -> None:
    profile["engine"] = _normalize_asr_engine_name(profile.get("engine") or "faster_whisper")
    profile.pop("funasr_online_mode", None)
    profile.pop("funasr", None)
    vad = profile.get("vad")
    if isinstance(vad, dict):
        vad["backend"] = _normalize_vad_backend_name(vad.get("backend"))


def _runtime_base_dirs() -> list[Path]:
    dirs: list[Path] = [Path.cwd()]

    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(item)
    return unique_dirs


def _resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in _runtime_base_dirs():
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def _resolve_write_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    existing = _resolve_existing_path(path)
    if existing.exists():
        return existing
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / path
    return Path.cwd() / path


def _ensure_config_file(config_path: str | Path = "config.yaml") -> Path:
    target = _resolve_write_path(config_path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _present_external_config_keys(AppConfig().to_dict())
    with target.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return target


def load_config(config_path: str | Path = "config.yaml", fallback_path: str | Path = "config.example.yaml") -> AppConfig:
    path = _resolve_existing_path(config_path)
    if not path.exists():
        fallback = _resolve_existing_path(fallback_path)
        if fallback.exists():
            path = fallback
        else:
            path = _ensure_config_file(config_path)

    with path.open("r", encoding="utf-8") as fp:
        raw: dict[str, Any] = yaml.safe_load(fp) or {}
    raw = _normalize_external_config_keys(raw)

    migrated = migrate_legacy_config(raw)
    config = AppConfig.from_dict(migrated)
    return config


def save_config(config: AppConfig, config_path: str | Path = "config.yaml") -> Path:
    path = _resolve_write_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _present_external_config_keys(config.to_dict())
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return path


def _normalize_external_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(raw)
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

        # 新增 vNext 欄位
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
        # Keep canonical vNext keys when writing config.
        asr_channels.pop("chinese", None)
        asr_channels.pop("english", None)
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
    # Drop deprecated aliases and derived channel configs; canonical external keys are tts + tts_channels.
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
        runtime.pop("config_schema_version", None)
        runtime.pop("last_migration_note", None)
    health_last_success = data.get("health_last_success")
    if isinstance(health_last_success, dict):
        if not any(str(value or "").strip() for value in health_last_success.values()):
            data.pop("health_last_success", None)
    return data


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not is_legacy_config(raw):
        return deepcopy(raw)

    result: dict[str, Any] = AppConfig().to_dict()

    audio = raw.get("audio") or {}
    result["audio"]["meeting_in"] = str(audio.get("meeting_in") or audio.get("remote_in") or "")
    result["audio"]["microphone_in"] = str(audio.get("microphone_in") or audio.get("local_mic_in") or "")
    result["audio"]["speaker_out"] = str(audio.get("speaker_out") or audio.get("local_tts_out") or "")
    result["audio"]["meeting_out"] = str(audio.get("meeting_out") or audio.get("meeting_tts_out") or "")

    mode = str((raw.get("direction") or {}).get("mode", "") or raw.get("session_mode", "meeting_to_local"))
    mode_map = {
        "remote_only": "meeting_to_local",
        "local_only": "local_to_meeting",
        "bidirectional": "bidirectional",
    }
    result["direction"]["mode"] = "bidirectional"

    language = raw.get("language") or {}
    result["language"]["meeting_source"] = str(language.get("meeting_source") or language.get("remote_source") or "en")
    result["language"]["meeting_target"] = str(
        language.get("meeting_target")
        or language.get("remote_target")
        or language.get("remote_translation_target")
        or "zh-TW"
    )
    result["language"]["local_source"] = str(language.get("local_source") or "zh-TW")
    result["language"]["local_target"] = str(
        language.get("local_target")
        or language.get("local_translation_target")
        or "en"
    )

    asr = raw.get("asr") or {}
    openai = raw.get("openai") or {}
    result["asr"]["model"] = str(asr.get("model") or openai.get("asr_model") or "large-v3-turbo")
    result["asr"]["device"] = str(asr.get("device") or "cuda")
    result["asr"]["compute_type"] = str(asr.get("compute_type") or "float16")
    result["asr"]["beam_size"] = int(asr.get("beam_size", 1))
    result["asr"]["final_beam_size"] = int(asr.get("final_beam_size", max(3, result["asr"]["beam_size"])))
    result["asr"]["condition_on_previous_text"] = bool(asr.get("condition_on_previous_text", True))
    result["asr"]["final_condition_on_previous_text"] = bool(asr.get("final_condition_on_previous_text", True))
    result["asr"]["initial_prompt"] = str(asr.get("initial_prompt", result["asr"]["initial_prompt"]))
    result["asr"]["hotwords"] = str(asr.get("hotwords", result["asr"]["hotwords"]))
    result["asr"]["speculative_draft_model"] = str(
        asr.get("speculative_draft_model", result["asr"]["speculative_draft_model"])
    )
    result["asr"]["speculative_num_beams"] = int(
        asr.get("speculative_num_beams", result["asr"]["speculative_num_beams"])
    )
    result["asr"]["temperature_fallback"] = str(asr.get("temperature_fallback", result["asr"]["temperature_fallback"]))
    result["asr"]["no_speech_threshold"] = float(asr.get("no_speech_threshold", result["asr"]["no_speech_threshold"]))
    if isinstance(asr.get("vad"), dict):
        result["asr"]["vad"].update(asr["vad"])
    result["asr"]["engine"] = _normalize_asr_engine_name(asr.get("engine") or result["asr"]["engine"])
    result["asr"]["vad"]["backend"] = _normalize_vad_backend_name(result["asr"]["vad"].get("backend"))
    if isinstance(asr.get("streaming"), dict):
        result["asr"]["streaming"].update(asr["streaming"])
    result["asr_channels"]["local"] = deepcopy(result["asr"])
    result["asr_channels"]["remote"] = deepcopy(result["asr"])
    for channel in ("local", "remote"):
        profile = result["asr_channels"][channel]
        profile["temperature_fallback"] = str(result["asr"]["temperature_fallback"])
        profile["final_beam_size"] = max(3, int(profile["beam_size"]))
        profile["final_condition_on_previous_text"] = bool(
            profile.get("final_condition_on_previous_text", result["asr"]["final_condition_on_previous_text"])
        )
        profile["initial_prompt"] = str(profile.get("initial_prompt", result["asr"]["initial_prompt"]))
        profile["hotwords"] = str(profile.get("hotwords", result["asr"]["hotwords"]))
        profile["speculative_draft_model"] = str(
            profile.get("speculative_draft_model", result["asr"]["speculative_draft_model"])
        )
        profile["speculative_num_beams"] = int(
            profile.get("speculative_num_beams", result["asr"]["speculative_num_beams"])
        )
        profile["no_speech_threshold"] = float(result["asr"]["no_speech_threshold"])
        profile["streaming"]["soft_final_audio_ms"] = int(result["asr"]["streaming"]["soft_final_audio_ms"])
        profile["engine"] = _normalize_asr_engine_name(profile.get("engine") or result["asr"]["engine"])
        if isinstance(profile.get("vad"), dict):
            profile["vad"]["backend"] = _normalize_vad_backend_name(profile["vad"].get("backend"))
        profile.pop("funasr_online_mode", None)
        profile.pop("funasr", None)

    llm = raw.get("llm") or {}
    result["llm"]["backend"] = "lm_studio"
    llm_base_url = str(llm.get("base_url", "")).strip()
    legacy_base_url = str(openai.get("base_url", "")).strip()
    if llm_base_url:
        result["llm"]["base_url"] = llm_base_url
    if legacy_base_url and "127.0.0.1" in legacy_base_url:
        result["llm"]["base_url"] = legacy_base_url
    if (not str(result["llm"]["base_url"]).strip()) or ("11434" in str(result["llm"]["base_url"])):
        result["llm"]["base_url"] = "http://127.0.0.1:1234"
    result["llm"]["model"] = str(llm.get("model") or openai.get("translate_model") or result["llm"]["model"])
    result["llm"]["temperature"] = float(llm.get("temperature", result["llm"]["temperature"]))
    result["llm"]["top_p"] = float(llm.get("top_p", result["llm"]["top_p"]))
    result["llm"]["max_output_tokens"] = int(llm.get("max_output_tokens", result["llm"]["max_output_tokens"]))
    result["llm"]["repeat_penalty"] = float(llm.get("repeat_penalty", result["llm"]["repeat_penalty"]))
    result["llm"]["stop_tokens"] = str(llm.get("stop_tokens", result["llm"]["stop_tokens"]))
    result["llm"]["request_timeout_sec"] = int(llm.get("request_timeout_sec", result["llm"]["request_timeout_sec"]))
    if isinstance(llm.get("sliding_window"), dict):
        result["llm"]["sliding_window"].update(llm["sliding_window"])
    if isinstance(llm.get("profiles"), dict):
        for key in ("live_caption_fast", "dialogue_fast", "live_caption_stable", "speech_output_natural", "technical_meeting"):
            profile_raw = (llm.get("profiles") or {}).get(key)
            if isinstance(profile_raw, dict):
                result["llm"]["profiles"][key].update(profile_raw)
    result["llm"]["caption_profile"] = str(llm.get("caption_profile", result["llm"]["caption_profile"]))
    result["llm"]["speech_profile"] = str(llm.get("speech_profile", result["llm"]["speech_profile"]))
    result["llm_channels"]["local"] = deepcopy(result["llm"])
    result["llm_channels"]["remote"] = deepcopy(result["llm"])

    tts = raw.get("tts") or {}
    result["tts"]["executable_path"] = str(tts.get("executable_path", result["tts"]["executable_path"]))
    result["tts"]["model_path"] = str(tts.get("model_path", result["tts"]["model_path"]))
    result["tts"]["config_path"] = str(tts.get("config_path", result["tts"]["config_path"]))
    result["tts"]["voice_name"] = str(tts.get("voice_name", result["tts"]["voice_name"]))
    result["tts"]["style_preset"] = str(tts.get("style_preset", result["tts"]["style_preset"]))
    result["tts"]["speaker_id"] = int(tts.get("speaker_id", result["tts"]["speaker_id"]))
    result["tts"]["length_scale"] = float(tts.get("length_scale", result["tts"]["length_scale"]))
    result["tts"]["noise_scale"] = float(tts.get("noise_scale", result["tts"]["noise_scale"]))
    result["tts"]["noise_w"] = float(tts.get("noise_w", result["tts"]["noise_w"]))
    result["tts"]["sample_rate"] = int(tts.get("sample_rate", result["tts"]["sample_rate"]))
    result["meeting_tts"] = deepcopy(result["tts"])
    result["local_tts"] = deepcopy(result["tts"])
    result["meeting_tts"]["voice_name"] = str(result["meeting_tts"].get("voice_name") or "zh-TW-HsiaoChenNeural")
    result["local_tts"]["voice_name"] = str(result["local_tts"].get("voice_name") or "en-US-JennyNeural")
    if str(result["local_tts"]["voice_name"]).strip().lower().startswith("zh-"):
        result["local_tts"]["voice_name"] = "en-US-JennyNeural"
    result["tts_channels"]["local"] = deepcopy(result["meeting_tts"])
    result["tts_channels"]["remote"] = deepcopy(result["local_tts"])

    runtime = raw.get("runtime") or {}
    result["runtime"]["sample_rate"] = int(runtime.get("sample_rate", raw.get("sample_rate", 48000)))
    result["runtime"]["chunk_ms"] = int(runtime.get("chunk_ms", raw.get("chunk_ms", 100)))
    result["runtime"]["passthrough_gain"] = float(runtime.get("passthrough_gain", result["runtime"]["passthrough_gain"]))
    result["runtime"]["tts_gain"] = float(runtime.get("tts_gain", result["runtime"]["tts_gain"]))
    result["runtime"]["latency_mode"] = str(runtime.get("latency_mode", result["runtime"]["latency_mode"]))
    result["runtime"]["display_partial_strategy"] = str(
        runtime.get("display_partial_strategy", result["runtime"]["display_partial_strategy"])
    )
    result["runtime"]["stable_partial_min_repeats"] = int(
        runtime.get("stable_partial_min_repeats", result["runtime"]["stable_partial_min_repeats"])
    )
    result["runtime"]["partial_stability_max_delta_chars"] = int(
        runtime.get("partial_stability_max_delta_chars", result["runtime"]["partial_stability_max_delta_chars"])
    )
    result["runtime"]["asr_partial_min_audio_ms"] = int(
        runtime.get("asr_partial_min_audio_ms", result["runtime"]["asr_partial_min_audio_ms"])
    )
    result["runtime"]["asr_partial_interval_floor_ms"] = int(
        runtime.get("asr_partial_interval_floor_ms", result["runtime"]["asr_partial_interval_floor_ms"])
    )
    result["runtime"]["llm_partial_interval_floor_ms"] = int(
        runtime.get("llm_partial_interval_floor_ms", result["runtime"]["llm_partial_interval_floor_ms"])
    )
    result["runtime"]["early_final_enabled"] = bool(
        runtime.get("early_final_enabled", result["runtime"]["early_final_enabled"])
    )
    result["runtime"]["tts_accept_stable_partial"] = bool(
        runtime.get("tts_accept_stable_partial", result["runtime"]["tts_accept_stable_partial"])
    )
    result["runtime"]["tts_partial_min_chars"] = int(
        runtime.get("tts_partial_min_chars", result["runtime"]["tts_partial_min_chars"])
    )
    result["runtime"]["tts_use_speech_profile"] = bool(
        runtime.get("tts_use_speech_profile", result["runtime"]["tts_use_speech_profile"])
    )
    asr_shared = int(runtime.get("asr_queue_maxsize", result["runtime"]["asr_queue_maxsize"]))
    llm_shared = int(runtime.get("llm_queue_maxsize", result["runtime"]["llm_queue_maxsize"]))
    tts_shared = int(runtime.get("tts_queue_maxsize", result["runtime"]["tts_queue_maxsize"]))
    result["runtime"]["asr_queue_maxsize_local"] = int(runtime.get("asr_queue_maxsize_local", asr_shared))
    result["runtime"]["asr_queue_maxsize_remote"] = int(runtime.get("asr_queue_maxsize_remote", asr_shared))
    result["runtime"]["llm_queue_maxsize_local"] = int(runtime.get("llm_queue_maxsize_local", llm_shared))
    result["runtime"]["llm_queue_maxsize_remote"] = int(runtime.get("llm_queue_maxsize_remote", llm_shared))
    result["runtime"]["tts_queue_maxsize_local"] = int(runtime.get("tts_queue_maxsize_local", tts_shared))
    result["runtime"]["tts_queue_maxsize_remote"] = int(runtime.get("tts_queue_maxsize_remote", tts_shared))
    result["runtime"]["asr_queue_maxsize"] = result["runtime"]["asr_queue_maxsize_local"]
    result["runtime"]["llm_queue_maxsize"] = result["runtime"]["llm_queue_maxsize_local"]
    result["runtime"]["tts_queue_maxsize"] = result["runtime"]["tts_queue_maxsize_local"]
    result["runtime"]["translation_exact_cache_size"] = int(
        runtime.get("translation_exact_cache_size", result["runtime"]["translation_exact_cache_size"])
    )
    result["runtime"]["translation_prefix_min_delta_chars"] = int(
        runtime.get("translation_prefix_min_delta_chars", result["runtime"]["translation_prefix_min_delta_chars"])
    )
    result["runtime"]["tts_cancel_pending_on_new_final"] = bool(
        runtime.get("tts_cancel_pending_on_new_final", result["runtime"]["tts_cancel_pending_on_new_final"])
    )
    result["runtime"]["tts_cancel_policy"] = str(runtime.get("tts_cancel_policy", result["runtime"]["tts_cancel_policy"]))
    result["runtime"]["tts_max_wait_ms"] = int(runtime.get("tts_max_wait_ms", result["runtime"]["tts_max_wait_ms"]))
    result["runtime"]["tts_max_chars"] = int(runtime.get("tts_max_chars", result["runtime"]["tts_max_chars"]))
    result["runtime"]["tts_drop_backlog_threshold"] = int(
        runtime.get("tts_drop_backlog_threshold", result["runtime"]["tts_drop_backlog_threshold"])
    )
    result["runtime"]["llm_streaming_tokens"] = int(runtime.get("llm_streaming_tokens", result["runtime"]["llm_streaming_tokens"]))
    result["runtime"]["max_pipeline_latency_ms"] = int(runtime.get("max_pipeline_latency_ms", result["runtime"]["max_pipeline_latency_ms"]))
    legacy_translation_enabled = bool(runtime.get("translation_enabled", True))
    result["runtime"]["remote_translation_enabled"] = bool(
        runtime.get("remote_translation_enabled", legacy_translation_enabled)
    )
    result["runtime"]["local_translation_enabled"] = bool(
        runtime.get("local_translation_enabled", legacy_translation_enabled)
    )
    result["runtime"]["translation_enabled"] = legacy_translation_enabled
    result["runtime"]["asr_language_mode"] = "auto"
    result["runtime"]["config_schema_version"] = 5
    result["runtime"]["last_migration_note"] = "migrated_from_legacy"
    result["runtime"]["warmup_on_start"] = False
    result["runtime"]["use_channel_specific_asr"] = True
    result["runtime"]["use_channel_specific_llm"] = True

    legacy_test = raw.get("provider_test_last_success") or {}
    health = raw.get("health_last_success") or {}
    result["health_last_success"]["asr"] = str(health.get("asr") or legacy_test.get("asr") or "")
    result["health_last_success"]["llm"] = str(health.get("llm") or legacy_test.get("translate") or "")
    result["health_last_success"]["tts"] = str(health.get("tts") or legacy_test.get("tts") or "")
    return result


def is_legacy_config(raw: dict[str, Any]) -> bool:
    if "openai" in raw or "model" in raw or "session_mode" in raw:
        return True
    has_asr = "asr" in raw or "asr_channels" in raw
    has_llm = "llm" in raw or "llm_channels" in raw
    if (not has_asr) or (not has_llm) or "tts" not in raw or "runtime" not in raw:
        return True
    audio = raw.get("audio") or {}
    if any(key in audio for key in ("remote_in", "local_mic_in", "local_tts_out", "meeting_tts_out")):
        return True
    return False

