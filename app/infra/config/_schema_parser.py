"""parse_app_config — extracted from AppConfig.from_dict to keep schema.py shorter."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from app.infra.config.schema import (
    AppConfig,
    AsrChannelsConfig,
    AsrConfig,
    AsrStreamingSettings,
    AudioRouteConfig,
    DirectionConfig,
    HealthStateConfig,
    LanguageConfig,
    LlmChannelsConfig,
    LlmConfig,
    LlmRuntimeConfig,
    RuntimeConfig,
    SlidingWindowConfig,
    TranslationProfileConfig,
    TranslationProfilesConfig,
    TtsChannelOverride,
    TtsChannelsConfig,
    TtsConfig,
    VadSettings,
    merge_tts_configs,
)


def _normalize_llm_backend(value: object) -> str:
    backend = str(value or "").strip().lower()
    return backend or "local_llama_inprocess"


def _parse_llm_runtime(raw: dict[str, Any], fallback: LlmRuntimeConfig | None = None) -> LlmRuntimeConfig:
    defaults = fallback or LlmRuntimeConfig()
    runtime_raw = raw.get("runtime") or {}
    if not isinstance(runtime_raw, dict):
        runtime_raw = {}
    return LlmRuntimeConfig(
        model_path=str(runtime_raw.get("model_path", defaults.model_path)),
        ctx_size=int(runtime_raw.get("ctx_size", defaults.ctx_size)),
        gpu_layers=int(runtime_raw.get("gpu_layers", defaults.gpu_layers)),
        threads=int(runtime_raw.get("threads", defaults.threads)),
        batch_size=int(runtime_raw.get("batch_size", defaults.batch_size)),
    )


def _parse_llm_config(raw: dict[str, Any], fallback: LlmConfig) -> LlmConfig:
    runtime = _parse_llm_runtime(raw, fallback.runtime)
    return LlmConfig(
        backend=_normalize_llm_backend(raw.get("backend", fallback.backend)),
        model=str(raw.get("model", fallback.model)) or fallback.model,
        runtime=runtime,
        temperature=float(raw.get("temperature", fallback.temperature)),
        top_p=float(raw.get("top_p", fallback.top_p)),
        max_output_tokens=int(raw.get("max_output_tokens", fallback.max_output_tokens)),
        repeat_penalty=float(raw.get("repeat_penalty", fallback.repeat_penalty)),
        stop_tokens=str(raw.get("stop_tokens", fallback.stop_tokens)),
        request_timeout_sec=int(raw.get("request_timeout_sec", fallback.request_timeout_sec)),
        sliding_window=SlidingWindowConfig(**(raw.get("sliding_window") or asdict(fallback.sliding_window))),
        profiles=TranslationProfilesConfig(
            live_caption_fast=TranslationProfileConfig(
                **((raw.get("profiles") or {}).get("live_caption_fast") or asdict(fallback.profiles.live_caption_fast))
            ),
            dialogue_fast=TranslationProfileConfig(
                **((raw.get("profiles") or {}).get("dialogue_fast") or asdict(fallback.profiles.dialogue_fast))
            ),
            live_caption_stable=TranslationProfileConfig(
                **((raw.get("profiles") or {}).get("live_caption_stable") or asdict(fallback.profiles.live_caption_stable))
            ),
            speech_output_natural=TranslationProfileConfig(
                **(
                    (raw.get("profiles") or {}).get("speech_output_natural")
                    or asdict(fallback.profiles.speech_output_natural)
                )
            ),
            technical_meeting=TranslationProfileConfig(
                **((raw.get("profiles") or {}).get("technical_meeting") or asdict(fallback.profiles.technical_meeting))
            ),
        ),
        caption_profile=str(raw.get("caption_profile", fallback.caption_profile)),
        speech_profile=str(raw.get("speech_profile", fallback.speech_profile)),
    )


def parse_app_config(raw: dict[str, Any]) -> AppConfig:  # noqa: PLR0912,PLR0915
    """Build a fully-validated :class:`AppConfig` from a raw dict."""
    _asr_d = AsrConfig()
    _llm_d = LlmConfig()

    audio = AudioRouteConfig(**(raw.get("audio") or {}))
    direction = DirectionConfig(**(raw.get("direction") or {}))
    language = LanguageConfig(**(raw.get("language") or {}))

    asr_raw = raw.get("asr") or {}
    asr = AsrConfig(
        engine=str(asr_raw.get("engine", _asr_d.engine)),
        model=str(asr_raw.get("model", _asr_d.model)),
        device=str(asr_raw.get("device", _asr_d.device)),
        compute_type=str(asr_raw.get("compute_type", _asr_d.compute_type)),
        beam_size=int(asr_raw.get("beam_size", _asr_d.beam_size)),
        final_beam_size=int(asr_raw.get("final_beam_size", max(_asr_d.final_beam_size, int(asr_raw.get("beam_size", _asr_d.beam_size))))),
        condition_on_previous_text=bool(asr_raw.get("condition_on_previous_text", _asr_d.condition_on_previous_text)),
        final_condition_on_previous_text=bool(asr_raw.get("final_condition_on_previous_text", _asr_d.final_condition_on_previous_text)),
        initial_prompt=str(asr_raw.get("initial_prompt", _asr_d.initial_prompt)),
        hotwords=str(asr_raw.get("hotwords", _asr_d.hotwords)),
        speculative_draft_model=str(asr_raw.get("speculative_draft_model", _asr_d.speculative_draft_model)),
        speculative_num_beams=int(asr_raw.get("speculative_num_beams", _asr_d.speculative_num_beams)),
        temperature_fallback=str(asr_raw.get("temperature_fallback", _asr_d.temperature_fallback)),
        no_speech_threshold=float(asr_raw.get("no_speech_threshold", _asr_d.no_speech_threshold)),
        hallucination_filter=bool(asr_raw.get("hallucination_filter", _asr_d.hallucination_filter)),
        vad=VadSettings(**(asr_raw.get("vad") or {})),
        streaming=AsrStreamingSettings(**(asr_raw.get("streaming") or {})),
    )
    asr_channels_raw = raw.get("asr_channels") or {}
    asr_local_raw = asr_channels_raw.get("local") or asr_raw
    asr_remote_raw = asr_channels_raw.get("remote") or asr_raw
    asr_channels = AsrChannelsConfig(
        local=AppConfig._parse_asr_config(asr_local_raw, asr),
        remote=AppConfig._parse_asr_config(asr_remote_raw, asr),
    )
    if not asr_raw and asr_channels_raw:
        asr = deepcopy(asr_channels.local)

    llm_raw = raw.get("llm") or {}
    llm = _parse_llm_config(llm_raw, _llm_d)
    llm_channels_raw = raw.get("llm_channels") or {}
    llm_local_raw = llm_channels_raw.get("local") or llm_raw
    llm_remote_raw = llm_channels_raw.get("remote") or llm_raw
    llm_channels = LlmChannelsConfig(
        local=_parse_llm_config(llm_local_raw, llm),
        remote=_parse_llm_config(llm_remote_raw, llm),
    )
    if not llm_raw and llm_channels_raw:
        llm = deepcopy(llm_channels.local)

    tts = TtsConfig(**(raw.get("tts") or {}))
    meeting_tts = TtsConfig(**(raw.get("meeting_tts") or raw.get("tts") or {}))
    local_tts = TtsConfig(**(raw.get("local_tts") or raw.get("tts") or {}))
    tts_channels_raw = raw.get("tts_channels") or {}
    tts_channels = TtsChannelsConfig(
        local=TtsChannelOverride(**(tts_channels_raw.get("local") or {})),
        remote=TtsChannelOverride(**(tts_channels_raw.get("remote") or {})),
    )
    if "meeting_tts" not in raw:
        meeting_tts = merge_tts_configs(tts, meeting_tts, tts_channels.local)
    if "local_tts" not in raw:
        local_tts = merge_tts_configs(tts, local_tts, tts_channels.remote)
    runtime = RuntimeConfig(**(raw.get("runtime") or {}))
    health_last_success = HealthStateConfig(**(raw.get("health_last_success") or {}))

    direction.mode = "bidirectional"
    for llm_config in (llm, llm_channels.local, llm_channels.remote):
        llm_config.backend = _normalize_llm_backend(llm_config.backend)
        if not llm_config.model.strip():
            llm_config.model = _llm_d.model
    valid_profiles = {
        "live_caption_fast",
        "dialogue_fast",
        "live_caption_stable",
        "speech_output_natural",
        "technical_meeting",
    }
    valid_tts_style_presets = {"balanced", "broadcast_clear", "conversational", "fast_response"}
    if llm.caption_profile not in valid_profiles:
        llm.caption_profile = "live_caption_fast"
    if llm.speech_profile not in valid_profiles:
        llm.speech_profile = "speech_output_natural"
    for tts_config in (tts, meeting_tts, local_tts):
        if (tts_config.style_preset or "").strip() not in valid_tts_style_presets:
            tts_config.style_preset = "balanced"
    if runtime.tts_output_mode not in {"subtitle_only", "tts", "passthrough"}:
        runtime.tts_output_mode = "subtitle_only"
    if runtime.asr_language_mode != "auto":
        runtime.asr_language_mode = "auto"
    runtime.use_channel_specific_asr = True
    runtime.use_channel_specific_llm = True

    # P1-4: 數值範圍驗證 — 夾緊而非拒絕，保持容錯
    runtime.passthrough_gain = max(0.0, min(4.0, float(runtime.passthrough_gain)))
    runtime.tts_gain = max(0.0, min(4.0, float(runtime.tts_gain)))
    runtime.chunk_ms = max(20, min(500, int(runtime.chunk_ms)))
    runtime.asr_queue_maxsize_local = max(8, int(runtime.asr_queue_maxsize_local))
    runtime.asr_queue_maxsize_remote = max(8, int(runtime.asr_queue_maxsize_remote))
    runtime.tts_queue_maxsize_local = max(4, int(runtime.tts_queue_maxsize_local))
    runtime.tts_queue_maxsize_remote = max(4, int(runtime.tts_queue_maxsize_remote))
    valid_display_partial = {"all", "none", "stable_only"}
    if runtime.display_partial_strategy not in valid_display_partial:
        runtime.display_partial_strategy = "stable_only"
    valid_asr_accuracy_modes = {"low_latency", "balanced", "high_accuracy"}
    if runtime.asr_accuracy_mode not in valid_asr_accuracy_modes:
        runtime.asr_accuracy_mode = "balanced"
    runtime.asr_final_rescue_max_attempts = max(0, min(3, int(runtime.asr_final_rescue_max_attempts)))
    runtime.asr_final_rescue_max_no_speech_prob = max(
        0.0,
        min(1.0, float(runtime.asr_final_rescue_max_no_speech_prob)),
    )
    runtime.asr_final_rescue_max_compression_ratio = max(
        1.0,
        float(runtime.asr_final_rescue_max_compression_ratio),
    )
    runtime.asr_final_rescue_min_chars = max(0, int(runtime.asr_final_rescue_min_chars))

    return AppConfig(
        audio=audio,
        direction=direction,
        language=language,
        asr=asr,
        asr_channels=asr_channels,
        llm=llm,
        llm_channels=llm_channels,
        tts=tts,
        meeting_tts=meeting_tts,
        local_tts=local_tts,
        tts_channels=tts_channels,
        runtime=runtime,
        health_last_success=health_last_success,
    )
