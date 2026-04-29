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
    llm = LlmConfig(
        backend=str(llm_raw.get("backend", _llm_d.backend)),
        base_url=str(llm_raw.get("base_url", _llm_d.base_url)),
        model=str(llm_raw.get("model", _llm_d.model)),
        temperature=float(llm_raw.get("temperature", _llm_d.temperature)),
        top_p=float(llm_raw.get("top_p", _llm_d.top_p)),
        max_output_tokens=int(llm_raw.get("max_output_tokens", _llm_d.max_output_tokens)),
        repeat_penalty=float(llm_raw.get("repeat_penalty", _llm_d.repeat_penalty)),
        stop_tokens=str(llm_raw.get("stop_tokens", _llm_d.stop_tokens)),
        request_timeout_sec=int(llm_raw.get("request_timeout_sec", _llm_d.request_timeout_sec)),
        sliding_window=SlidingWindowConfig(**(llm_raw.get("sliding_window") or {})),
        profiles=TranslationProfilesConfig(
            live_caption_fast=TranslationProfileConfig(
                **((llm_raw.get("profiles") or {}).get("live_caption_fast") or {})
            ),
            dialogue_fast=TranslationProfileConfig(
                **((llm_raw.get("profiles") or {}).get("dialogue_fast") or {})
            ),
            live_caption_stable=TranslationProfileConfig(
                **((llm_raw.get("profiles") or {}).get("live_caption_stable") or {})
            ),
            speech_output_natural=TranslationProfileConfig(
                **((llm_raw.get("profiles") or {}).get("speech_output_natural") or {})
            ),
            technical_meeting=TranslationProfileConfig(
                **((llm_raw.get("profiles") or {}).get("technical_meeting") or {})
            ),
        ),
        caption_profile=str(llm_raw.get("caption_profile", _llm_d.caption_profile)),
        speech_profile=str(llm_raw.get("speech_profile", _llm_d.speech_profile)),
    )
    llm_channels_raw = raw.get("llm_channels") or {}
    llm_local_raw = llm_channels_raw.get("local") or llm_raw
    llm_remote_raw = llm_channels_raw.get("remote") or llm_raw
    llm_channels = LlmChannelsConfig(
        local=LlmConfig(
            backend=str(llm_local_raw.get("backend", llm.backend)),
            base_url=str(llm_local_raw.get("base_url", llm.base_url)),
            model=str(llm_local_raw.get("model", llm.model)),
            temperature=float(llm_local_raw.get("temperature", llm.temperature)),
            top_p=float(llm_local_raw.get("top_p", llm.top_p)),
            max_output_tokens=int(llm_local_raw.get("max_output_tokens", llm.max_output_tokens)),
            repeat_penalty=float(llm_local_raw.get("repeat_penalty", llm.repeat_penalty)),
            stop_tokens=str(llm_local_raw.get("stop_tokens", llm.stop_tokens)),
            request_timeout_sec=int(llm_local_raw.get("request_timeout_sec", llm.request_timeout_sec)),
            sliding_window=SlidingWindowConfig(**(llm_local_raw.get("sliding_window") or asdict(llm.sliding_window))),
            profiles=TranslationProfilesConfig(
                live_caption_fast=TranslationProfileConfig(
                    **((llm_local_raw.get("profiles") or {}).get("live_caption_fast") or asdict(llm.profiles.live_caption_fast))
                ),
                dialogue_fast=TranslationProfileConfig(
                    **((llm_local_raw.get("profiles") or {}).get("dialogue_fast") or asdict(llm.profiles.dialogue_fast))
                ),
                live_caption_stable=TranslationProfileConfig(
                    **((llm_local_raw.get("profiles") or {}).get("live_caption_stable") or asdict(llm.profiles.live_caption_stable))
                ),
                speech_output_natural=TranslationProfileConfig(
                    **(
                        (llm_local_raw.get("profiles") or {}).get("speech_output_natural")
                        or asdict(llm.profiles.speech_output_natural)
                    )
                ),
                technical_meeting=TranslationProfileConfig(
                    **((llm_local_raw.get("profiles") or {}).get("technical_meeting") or asdict(llm.profiles.technical_meeting))
                ),
            ),
            caption_profile=str(llm_local_raw.get("caption_profile", llm.caption_profile)),
            speech_profile=str(llm_local_raw.get("speech_profile", llm.speech_profile)),
        ),
        remote=LlmConfig(
            backend=str(llm_remote_raw.get("backend", llm.backend)),
            base_url=str(llm_remote_raw.get("base_url", llm.base_url)),
            model=str(llm_remote_raw.get("model", llm.model)),
            temperature=float(llm_remote_raw.get("temperature", llm.temperature)),
            top_p=float(llm_remote_raw.get("top_p", llm.top_p)),
            max_output_tokens=int(llm_remote_raw.get("max_output_tokens", llm.max_output_tokens)),
            repeat_penalty=float(llm_remote_raw.get("repeat_penalty", llm.repeat_penalty)),
            stop_tokens=str(llm_remote_raw.get("stop_tokens", llm.stop_tokens)),
            request_timeout_sec=int(llm_remote_raw.get("request_timeout_sec", llm.request_timeout_sec)),
            sliding_window=SlidingWindowConfig(
                **(llm_remote_raw.get("sliding_window") or asdict(llm.sliding_window))
            ),
            profiles=TranslationProfilesConfig(
                live_caption_fast=TranslationProfileConfig(
                    **((llm_remote_raw.get("profiles") or {}).get("live_caption_fast") or asdict(llm.profiles.live_caption_fast))
                ),
                dialogue_fast=TranslationProfileConfig(
                    **((llm_remote_raw.get("profiles") or {}).get("dialogue_fast") or asdict(llm.profiles.dialogue_fast))
                ),
                live_caption_stable=TranslationProfileConfig(
                    **((llm_remote_raw.get("profiles") or {}).get("live_caption_stable") or asdict(llm.profiles.live_caption_stable))
                ),
                speech_output_natural=TranslationProfileConfig(
                    **(
                        (llm_remote_raw.get("profiles") or {}).get("speech_output_natural")
                        or asdict(llm.profiles.speech_output_natural)
                    )
                ),
                technical_meeting=TranslationProfileConfig(
                    **((llm_remote_raw.get("profiles") or {}).get("technical_meeting") or asdict(llm.profiles.technical_meeting))
                ),
            ),
            caption_profile=str(llm_remote_raw.get("caption_profile", llm.caption_profile)),
            speech_profile=str(llm_remote_raw.get("speech_profile", llm.speech_profile)),
        ),
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
    if llm.backend != "lm_studio":
        llm.backend = "lm_studio"
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
    if (not llm.base_url.strip()) or ("11434" in llm.base_url):
        llm.base_url = "http://127.0.0.1:1234"
    for tts_config in (tts, meeting_tts, local_tts):
        if (tts_config.style_preset or "").strip() not in valid_tts_style_presets:
            tts_config.style_preset = "balanced"
    if runtime.tts_output_mode not in {"subtitle_only", "tts", "passthrough"}:
        runtime.tts_output_mode = "subtitle_only"
    if runtime.asr_language_mode != "auto":
        runtime.asr_language_mode = "auto"
    runtime.use_channel_specific_asr = True
    runtime.use_channel_specific_llm = True

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
