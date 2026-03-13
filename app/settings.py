from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from app.schemas import AppConfig


def load_config(config_path: str | Path = "config.yaml", fallback_path: str | Path = "config.example.yaml") -> AppConfig:
    path = Path(config_path)
    from_fallback = False
    if not path.exists():
        path = Path(fallback_path)
        from_fallback = True
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        raw: dict[str, Any] = yaml.safe_load(fp) or {}

    migrated = migrate_legacy_config(raw)
    config = AppConfig.from_dict(migrated)
    return config


def save_config(config: AppConfig, config_path: str | Path = "config.yaml") -> Path:
    path = Path(config_path)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config.to_dict(), fp, sort_keys=False, allow_unicode=True)
    return path


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
    result["direction"]["mode"] = mode_map.get(mode, mode if mode in mode_map.values() else "meeting_to_local")

    language = raw.get("language") or {}
    result["language"]["meeting_source"] = str(language.get("meeting_source") or language.get("remote_source") or "en")
    result["language"]["meeting_target"] = str(language.get("meeting_target") or language.get("remote_target") or "zh-TW")
    result["language"]["local_source"] = str(language.get("local_source") or "zh-TW")
    result["language"]["local_target"] = str(language.get("local_target") or "en")

    asr = raw.get("asr") or {}
    openai = raw.get("openai") or {}
    result["asr"]["model"] = str(asr.get("model") or openai.get("asr_model") or "large-v3")
    result["asr"]["device"] = str(asr.get("device") or "cuda")
    result["asr"]["compute_type"] = str(asr.get("compute_type") or "float16")
    result["asr"]["beam_size"] = int(asr.get("beam_size", 1))
    result["asr"]["condition_on_previous_text"] = bool(asr.get("condition_on_previous_text", True))
    if isinstance(asr.get("vad"), dict):
        result["asr"]["vad"].update(asr["vad"])
    if isinstance(asr.get("streaming"), dict):
        result["asr"]["streaming"].update(asr["streaming"])

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
    result["llm"]["request_timeout_sec"] = int(llm.get("request_timeout_sec", result["llm"]["request_timeout_sec"]))
    if isinstance(llm.get("sliding_window"), dict):
        result["llm"]["sliding_window"].update(llm["sliding_window"])
    if isinstance(llm.get("profiles"), dict):
        for key in ("live_caption_fast", "live_caption_stable", "speech_output_natural", "technical_meeting"):
            profile_raw = (llm.get("profiles") or {}).get(key)
            if isinstance(profile_raw, dict):
                result["llm"]["profiles"][key].update(profile_raw)
    result["llm"]["caption_profile"] = str(llm.get("caption_profile", result["llm"]["caption_profile"]))
    result["llm"]["speech_profile"] = str(llm.get("speech_profile", result["llm"]["speech_profile"]))

    tts = raw.get("tts") or {}
    result["tts"]["executable_path"] = str(tts.get("executable_path", result["tts"]["executable_path"]))
    result["tts"]["model_path"] = str(tts.get("model_path", result["tts"]["model_path"]))
    result["tts"]["config_path"] = str(tts.get("config_path", result["tts"]["config_path"]))
    result["tts"]["voice_name"] = str(tts.get("voice_name", result["tts"]["voice_name"]))
    result["tts"]["speaker_id"] = int(tts.get("speaker_id", result["tts"]["speaker_id"]))
    result["tts"]["length_scale"] = float(tts.get("length_scale", result["tts"]["length_scale"]))
    result["tts"]["noise_scale"] = float(tts.get("noise_scale", result["tts"]["noise_scale"]))
    result["tts"]["noise_w"] = float(tts.get("noise_w", result["tts"]["noise_w"]))
    result["tts"]["sample_rate"] = int(tts.get("sample_rate", result["tts"]["sample_rate"]))
    result["meeting_tts"] = deepcopy(result["tts"])
    result["local_tts"] = deepcopy(result["tts"])
    result["tts_channels"]["local"] = deepcopy(result["meeting_tts"])
    result["tts_channels"]["remote"] = deepcopy(result["local_tts"])

    runtime = raw.get("runtime") or {}
    result["runtime"]["sample_rate"] = int(runtime.get("sample_rate", raw.get("sample_rate", 48000)))
    result["runtime"]["chunk_ms"] = int(runtime.get("chunk_ms", raw.get("chunk_ms", 100)))
    result["runtime"]["asr_queue_maxsize"] = int(runtime.get("asr_queue_maxsize", result["runtime"]["asr_queue_maxsize"]))
    result["runtime"]["llm_queue_maxsize"] = int(runtime.get("llm_queue_maxsize", result["runtime"]["llm_queue_maxsize"]))
    result["runtime"]["tts_queue_maxsize"] = int(runtime.get("tts_queue_maxsize", result["runtime"]["tts_queue_maxsize"]))
    result["runtime"]["translation_exact_cache_size"] = int(
        runtime.get("translation_exact_cache_size", result["runtime"]["translation_exact_cache_size"])
    )
    result["runtime"]["translation_prefix_min_delta_chars"] = int(
        runtime.get("translation_prefix_min_delta_chars", result["runtime"]["translation_prefix_min_delta_chars"])
    )
    result["runtime"]["tts_cancel_pending_on_new_final"] = bool(
        runtime.get("tts_cancel_pending_on_new_final", result["runtime"]["tts_cancel_pending_on_new_final"])
    )
    result["runtime"]["tts_drop_backlog_threshold"] = int(
        runtime.get("tts_drop_backlog_threshold", result["runtime"]["tts_drop_backlog_threshold"])
    )
    result["runtime"]["warmup_on_start"] = bool(runtime.get("warmup_on_start", result["runtime"]["warmup_on_start"]))

    legacy_test = raw.get("provider_test_last_success") or {}
    health = raw.get("health_last_success") or {}
    result["health_last_success"]["asr"] = str(health.get("asr") or legacy_test.get("asr") or "")
    result["health_last_success"]["llm"] = str(health.get("llm") or legacy_test.get("translate") or "")
    result["health_last_success"]["tts"] = str(health.get("tts") or legacy_test.get("tts") or "")
    return result


def is_legacy_config(raw: dict[str, Any]) -> bool:
    if "openai" in raw or "model" in raw or "session_mode" in raw:
        return True
    if (
        "direction" not in raw
        or "asr" not in raw
        or "llm" not in raw
        or "tts" not in raw
        or "runtime" not in raw
    ):
        return True
    audio = raw.get("audio") or {}
    if any(key in audio for key in ("remote_in", "local_mic_in", "local_tts_out", "meeting_tts_out")):
        return True
    return False
