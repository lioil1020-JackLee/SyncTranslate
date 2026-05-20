"""Legacy config migration helpers — extracted from settings_store."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

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


def is_legacy_config(raw: dict[str, Any]) -> bool:
    if "openai" in raw or "model" in raw or "session_mode" in raw:
        return True
    has_asr = "asr" in raw or "asr_channels" in raw or "asr_profiles" in raw
    has_llm = "llm" in raw or "llm_channels" in raw
    if (not has_asr) or (not has_llm) or "tts" not in raw or "runtime" not in raw:
        return True
    audio = raw.get("audio") or {}
    if any(key in audio for key in ("remote_in", "local_mic_in", "local_tts_out", "meeting_tts_out")):
        return True
    return False


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not is_legacy_config(raw):
        return deepcopy(raw)

    result: dict[str, Any] = AppConfig().to_dict()

    audio = raw.get("audio") or {}
    result["audio"]["meeting_in"] = str(audio.get("meeting_in") or audio.get("remote_in") or "")
    result["audio"]["microphone_in"] = str(audio.get("microphone_in") or audio.get("local_mic_in") or "")
    result["audio"]["speaker_out"] = str(audio.get("speaker_out") or audio.get("local_tts_out") or "")
    result["audio"]["meeting_out"] = str(audio.get("meeting_out") or audio.get("meeting_tts_out") or "")
    
    # Phase 7 routing_mode upgrade: old manual/third-party virtual audio setups
    # are normalized to synctranslate_virtual_audio by AppConfig.from_dict().
    old_routing_mode = str(audio.get("routing_mode", "")).strip().lower()
    if old_routing_mode == "advanced_manual":
        # Keep the legacy value only in the intermediate migration payload so
        # downstream parsing can perform one canonical normalization step.
        result["audio"]["routing_mode"] = "advanced_manual"
    else:
        # All other cases → new standard synctranslate_virtual_audio mode
        result["audio"]["routing_mode"] = "synctranslate_virtual_audio"
        result["runtime"]["last_migration_note"] = "upgraded_to_virtual_audio_mode"

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
    result["llm"]["backend"] = "local_llama_inprocess"
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
    result["runtime"]["config_schema_version"] = 6
    # Set migration note if not already set (routing_mode migration may have set it)
    if result["runtime"].get("last_migration_note") != "upgraded_to_virtual_audio_mode":
        result["runtime"]["last_migration_note"] = "migrated_from_legacy"
    result["runtime"]["warmup_on_start"] = True
    result["runtime"]["use_channel_specific_asr"] = True
    result["runtime"]["use_channel_specific_llm"] = True

    legacy_test = raw.get("provider_test_last_success") or {}
    health = raw.get("health_last_success") or {}
    result["health_last_success"]["asr"] = str(health.get("asr") or legacy_test.get("asr") or "")
    result["health_last_success"]["llm"] = str(health.get("llm") or legacy_test.get("translate") or "")
    result["health_last_success"]["tts"] = str(health.get("tts") or legacy_test.get("tts") or "")
    return result
