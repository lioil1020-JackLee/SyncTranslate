from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AudioRouteConfig:
    meeting_in: str = ""
    microphone_in: str = ""
    speaker_out: str = ""
    meeting_out: str = ""
    meeting_in_gain: float = 1.0
    microphone_in_gain: float = 1.0
    speaker_out_volume: float = 1.0
    meeting_out_volume: float = 1.0


@dataclass(slots=True)
class DirectionConfig:
    mode: str = "bidirectional"


@dataclass(slots=True)
class LanguageConfig:
    meeting_source: str = "en"
    meeting_target: str = "zh-TW"
    local_source: str = "zh-TW"
    local_target: str = "en"


@dataclass(slots=True)
class VadSettings:
    enabled: bool = True
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 520
    max_speech_duration_s: float = 10.0
    speech_pad_ms: int = 450
    rms_threshold: float = 0.02


@dataclass(slots=True)
class AsrStreamingSettings:
    partial_interval_ms: int = 800
    partial_history_seconds: int = 3
    final_history_seconds: int = 6
    soft_final_audio_ms: int = 4200


@dataclass(slots=True)
class AsrConfig:
    engine: str = "faster_whisper"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 3
    final_beam_size: int = 3
    condition_on_previous_text: bool = True
    final_condition_on_previous_text: bool = False
    temperature_fallback: str = "0.0,0.2,0.4"
    no_speech_threshold: float = 0.55
    vad: VadSettings = field(default_factory=VadSettings)
    streaming: AsrStreamingSettings = field(default_factory=AsrStreamingSettings)


@dataclass(slots=True)
class AsrChannelsConfig:
    local: AsrConfig = field(default_factory=AsrConfig)
    remote: AsrConfig = field(default_factory=AsrConfig)


@dataclass(slots=True)
class SlidingWindowConfig:
    enabled: bool = True
    trigger_tokens: int = 20
    max_context_items: int = 6


@dataclass(slots=True)
class TranslationProfileConfig:
    name: str = "live_caption_fast"
    prompt_style: str = "literal"
    context_items: int = 4
    partial_trigger_tokens: int = 18
    max_tokens: int = 256
    preserve_terms: bool = True
    naturalize_tone: bool = False
    allow_subject_completion: bool = False


@dataclass(slots=True)
class TranslationProfilesConfig:
    live_caption_fast: TranslationProfileConfig = field(
        default_factory=lambda: TranslationProfileConfig(
            name="live_caption_fast",
            prompt_style="literal",
            context_items=4,
            partial_trigger_tokens=16,
            max_tokens=192,
            preserve_terms=True,
            naturalize_tone=False,
            allow_subject_completion=False,
        )
    )
    live_caption_stable: TranslationProfileConfig = field(
        default_factory=lambda: TranslationProfileConfig(
            name="live_caption_stable",
            prompt_style="stable",
            context_items=8,
            partial_trigger_tokens=20,
            max_tokens=256,
            preserve_terms=True,
            naturalize_tone=False,
            allow_subject_completion=False,
        )
    )
    speech_output_natural: TranslationProfileConfig = field(
        default_factory=lambda: TranslationProfileConfig(
            name="speech_output_natural",
            prompt_style="natural",
            context_items=6,
            partial_trigger_tokens=24,
            max_tokens=256,
            preserve_terms=True,
            naturalize_tone=True,
            allow_subject_completion=True,
        )
    )
    technical_meeting: TranslationProfileConfig = field(
        default_factory=lambda: TranslationProfileConfig(
            name="technical_meeting",
            prompt_style="technical",
            context_items=8,
            partial_trigger_tokens=22,
            max_tokens=320,
            preserve_terms=True,
            naturalize_tone=False,
            allow_subject_completion=False,
        )
    )


@dataclass(slots=True)
class LlmConfig:
    backend: str = "lm_studio"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen/qwen3.5-9b"
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 128
    repeat_penalty: float = 1.05
    stop_tokens: str = "</target>\nTranslation:"
    request_timeout_sec: int = 20
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    profiles: TranslationProfilesConfig = field(default_factory=TranslationProfilesConfig)
    caption_profile: str = "live_caption_fast"
    speech_profile: str = "speech_output_natural"


@dataclass(slots=True)
class LlmChannelsConfig:
    local: LlmConfig = field(default_factory=LlmConfig)
    remote: LlmConfig = field(default_factory=LlmConfig)


@dataclass(slots=True)
class TtsConfig:
    engine: str = "edge_tts"
    executable_path: str = ""
    model_path: str = ""
    config_path: str = ""
    voice_name: str = "zh-TW-HsiaoChenNeural"
    style_preset: str = "balanced"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sample_rate: int = 22050


@dataclass(slots=True)
class TtsChannelOverride:
    engine: str | None = None
    executable_path: str | None = None
    model_path: str | None = None
    config_path: str | None = None
    voice_name: str | None = None
    style_preset: str | None = None
    speaker_id: int | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w: float | None = None
    sample_rate: int | None = None


@dataclass(slots=True)
class TtsChannelsConfig:
    local: TtsChannelOverride = field(default_factory=TtsChannelOverride)
    remote: TtsChannelOverride = field(default_factory=TtsChannelOverride)


@dataclass(slots=True)
class RuntimeConfig:
    sample_rate: int = 48000
    chunk_ms: int = 100
    passthrough_gain: float = 1.6
    tts_gain: float = 1.4
    asr_pre_roll_ms: int = 500
    latency_mode: str = "balanced"
    display_partial_strategy: str = "stable_only"
    stable_partial_min_repeats: int = 2
    partial_stability_max_delta_chars: int = 8
    asr_partial_min_audio_ms: int = 280
    asr_partial_interval_floor_ms: int = 280
    llm_partial_interval_floor_ms: int = 320
    early_final_enabled: bool = True
    tts_accept_stable_partial: bool = True
    tts_partial_min_chars: int = 12
    tts_use_speech_profile: bool = False
    asr_queue_maxsize_local: int = 128
    asr_queue_maxsize_remote: int = 128
    llm_queue_maxsize_local: int = 32
    llm_queue_maxsize_remote: int = 32
    tts_queue_maxsize_local: int = 32
    tts_queue_maxsize_remote: int = 32
    # Legacy shared queue settings kept for backward compatibility.
    asr_queue_maxsize: int = 128
    llm_queue_maxsize: int = 32
    tts_queue_maxsize: int = 32
    translation_exact_cache_size: int = 256
    translation_prefix_min_delta_chars: int = 6
    tts_cancel_pending_on_new_final: bool = True
    tts_cancel_policy: str = "all_pending"
    tts_max_wait_ms: int = 4000
    tts_max_chars: int = 200
    tts_drop_backlog_threshold: int = 6
    llm_streaming_tokens: int = 16
    max_pipeline_latency_ms: int = 3000
    local_echo_guard_enabled: bool = False
    local_echo_guard_resume_delay_ms: int = 300
    remote_echo_guard_resume_delay_ms: int = 300
    config_schema_version: int = 5
    last_migration_note: str = ""
    warmup_on_start: bool = False
    remote_translation_enabled: bool = True
    local_translation_enabled: bool = True
    translation_enabled: bool = True
    tts_output_mode: str = "subtitle_only"
    asr_language_mode: str = "auto"
    # 新增控制項：手動固定 ASR 來源語言（auto / zh-TW / en / ja / ko / th）
    remote_asr_language: str = "auto"
    local_asr_language: str = "auto"
    # 新增控制項：翻譯目標語言（none / zh-TW / en / ja / ko / th）
    remote_translation_target: str = "zh-TW"
    local_translation_target: str = "en"
    # 新增控制項：TTS 聲線（none 表示不輸出）
    remote_tts_voice: str = ""
    local_tts_voice: str = ""
    remote_tts_enabled: bool = False
    local_tts_enabled: bool = False
    use_channel_specific_asr: bool = True
    use_channel_specific_llm: bool = True


@dataclass(slots=True)
class HealthStateConfig:
    asr: str = ""
    llm: str = ""
    tts: str = ""


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    hostapi_index: int
    hostapi_name: str
    hostapi_label: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


@dataclass(slots=True)
class AppConfig:
    audio: AudioRouteConfig = field(default_factory=AudioRouteConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    asr_channels: AsrChannelsConfig = field(default_factory=AsrChannelsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    llm_channels: LlmChannelsConfig = field(default_factory=LlmChannelsConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    meeting_tts: TtsConfig = field(default_factory=TtsConfig)
    local_tts: TtsConfig = field(default_factory=TtsConfig)
    tts_channels: TtsChannelsConfig = field(default_factory=TtsChannelsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    health_last_success: HealthStateConfig = field(default_factory=HealthStateConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        # Sentinel defaults — all fallback values derive from these instances,
        # which are the canonical default source for config fallback behavior.
        # Keep default values aligned at dataclass field definitions.
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
            temperature_fallback=str(asr_raw.get("temperature_fallback", _asr_d.temperature_fallback)),
            no_speech_threshold=float(asr_raw.get("no_speech_threshold", _asr_d.no_speech_threshold)),
            vad=VadSettings(**(asr_raw.get("vad") or {})),
            streaming=AsrStreamingSettings(**(asr_raw.get("streaming") or {})),
        )
        asr_channels_raw = raw.get("asr_channels") or {}
        asr_local_raw = asr_channels_raw.get("local") or asr_raw
        asr_remote_raw = asr_channels_raw.get("remote") or asr_raw
        asr_channels = AsrChannelsConfig(
            local=AsrConfig(
                engine=str(asr_local_raw.get("engine", asr.engine)),
                model=str(asr_local_raw.get("model", asr.model)),
                device=str(asr_local_raw.get("device", asr.device)),
                compute_type=str(asr_local_raw.get("compute_type", asr.compute_type)),
                beam_size=int(asr_local_raw.get("beam_size", asr.beam_size)),
                final_beam_size=int(asr_local_raw.get("final_beam_size", asr.final_beam_size)),
                condition_on_previous_text=bool(
                    asr_local_raw.get("condition_on_previous_text", asr.condition_on_previous_text)
                ),
                final_condition_on_previous_text=bool(
                    asr_local_raw.get(
                        "final_condition_on_previous_text",
                        asr.final_condition_on_previous_text,
                    )
                ),
                temperature_fallback=str(asr_local_raw.get("temperature_fallback", asr.temperature_fallback)),
                no_speech_threshold=float(asr_local_raw.get("no_speech_threshold", asr.no_speech_threshold)),
                vad=VadSettings(**(asr_local_raw.get("vad") or asdict(asr.vad))),
                streaming=AsrStreamingSettings(**(asr_local_raw.get("streaming") or asdict(asr.streaming))),
            ),
            remote=AsrConfig(
                engine=str(asr_remote_raw.get("engine", asr.engine)),
                model=str(asr_remote_raw.get("model", asr.model)),
                device=str(asr_remote_raw.get("device", asr.device)),
                compute_type=str(asr_remote_raw.get("compute_type", asr.compute_type)),
                beam_size=int(asr_remote_raw.get("beam_size", asr.beam_size)),
                final_beam_size=int(asr_remote_raw.get("final_beam_size", asr.final_beam_size)),
                condition_on_previous_text=bool(
                    asr_remote_raw.get("condition_on_previous_text", asr.condition_on_previous_text)
                ),
                final_condition_on_previous_text=bool(
                    asr_remote_raw.get(
                        "final_condition_on_previous_text",
                        asr.final_condition_on_previous_text,
                    )
                ),
                temperature_fallback=str(asr_remote_raw.get("temperature_fallback", asr.temperature_fallback)),
                no_speech_threshold=float(asr_remote_raw.get("no_speech_threshold", asr.no_speech_threshold)),
                vad=VadSettings(**(asr_remote_raw.get("vad") or asdict(asr.vad))),
                streaming=AsrStreamingSettings(**(asr_remote_raw.get("streaming") or asdict(asr.streaming))),
            ),
        )

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
        valid_profiles = {"live_caption_fast", "live_caption_stable", "speech_output_natural", "technical_meeting"}
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

        return cls(
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_tts_configs(base: TtsConfig, fallback: TtsConfig, override: TtsChannelOverride) -> TtsConfig:
    return TtsConfig(
        engine=(override.engine if override.engine is not None else fallback.engine) or base.engine,
        executable_path=(override.executable_path if override.executable_path is not None else fallback.executable_path)
        or base.executable_path,
        model_path=(override.model_path if override.model_path is not None else fallback.model_path) or base.model_path,
        config_path=(override.config_path if override.config_path is not None else fallback.config_path)
        or base.config_path,
        voice_name=(override.voice_name if override.voice_name is not None else fallback.voice_name) or base.voice_name,
        style_preset=(override.style_preset if override.style_preset is not None else fallback.style_preset)
        or base.style_preset,
        speaker_id=(override.speaker_id if override.speaker_id is not None else fallback.speaker_id),
        length_scale=(override.length_scale if override.length_scale is not None else fallback.length_scale),
        noise_scale=(override.noise_scale if override.noise_scale is not None else fallback.noise_scale),
        noise_w=(override.noise_w if override.noise_w is not None else fallback.noise_w),
        sample_rate=(override.sample_rate if override.sample_rate is not None else fallback.sample_rate),
    )


def translation_enabled_for_source(runtime: RuntimeConfig, source: str) -> bool:
    legacy_enabled = bool(getattr(runtime, "translation_enabled", True))
    if source == "remote":
        return bool(getattr(runtime, "remote_translation_enabled", legacy_enabled))
    if source == "local":
        return bool(getattr(runtime, "local_translation_enabled", legacy_enabled))
    return legacy_enabled
