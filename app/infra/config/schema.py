from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from app.domain.constants import (
    ASR_DEFAULT_CHUNK_MS,
    ASR_DEFAULT_FRONTEND_HIGHPASS_ALPHA,
    TTS_DEFAULT_DROP_BACKLOG_THRESHOLD,
    TTS_DEFAULT_MAX_CHARS,
    TTS_DEFAULT_MAX_WAIT_MS,
    TTS_DEFAULT_PARTIAL_MIN_CHARS,
    TTS_DEFAULT_QUEUE_MAXSIZE,
)


DEFAULT_FIXED_LLM_MODEL = "hy-mt1.5-7b"


@dataclass(slots=True)
class AudioSystemDeviceConfig:
    capture_role: str = "communications"
    render_role: str = "communications"
    follow_system_default: bool = True
    fallback_to_multimedia_role: bool = True
    exclude_virtual_devices: bool = True


@dataclass(slots=True)
class VirtualAudioConfig:
    speaker_name: str = "SyncTranslate Virtual Speaker"
    microphone_name: str = "SyncTranslate Virtual Microphone"
    bridge_enabled: bool = True
    bridge_path: str = r"runtimes\audio\sync_audio_bridge.exe"
    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 16
    dtype: str = "int16"
    frame_ms: int = 10
    target_latency_ms: int = 120
    require_driver: bool = True


@dataclass(slots=True)
class CallTranslationConfig:
    listen_remote_original: bool = True
    listen_remote_translation: bool = True
    output_local_original: bool = False
    output_local_translation: bool = True


@dataclass(slots=True)
class AudioRouteConfig:
    meeting_in: str = ""
    microphone_in: str = ""
    speaker_out: str = ""
    meeting_out: str = ""
    routing_mode: str = "synctranslate_virtual_audio"
    system_devices: AudioSystemDeviceConfig = field(default_factory=AudioSystemDeviceConfig)
    virtual_audio: VirtualAudioConfig = field(default_factory=VirtualAudioConfig)
    call_translation: CallTranslationConfig = field(default_factory=CallTranslationConfig)


@dataclass(slots=True)
class WindowsIoConfig:
    preferred_sample_rate: int = 48000
    preferred_channels: int = 2
    internal_dtype: str = "float32"


@dataclass(slots=True)
class MeetingModeConfig:
    audio_source: str = "system_input"
    input_device: str = ""
    output_loopback_device: str = ""
    asr_language: str = "zh-TW"
    translation_target: str = "en"
    record_transcript: bool = True
    tts_enabled: bool = False

    @property
    def source_kind(self) -> str:
        return self.audio_source

    @property
    def source_device(self) -> str:
        return self.output_loopback_device if self.audio_source == "system_output_loopback" else self.input_device


@dataclass(slots=True)
class DialogueDirectionConfig:
    asr_language: str = "en"
    translation_target: str = "zh-TW"
    tts_voice: str = "none"
    output_policy: str = "direct_passthrough"
    passthrough_gain: float = 1.0


@dataclass(slots=True)
class DialogueModeConfig:
    remote_asr_language: str = "en"
    local_asr_language: str = "zh-TW"
    remote_translation_target: str = "zh-TW"
    local_translation_target: str = "en"
    remote_tts_voice: str = "none"
    local_tts_voice: str = "none"
    passthrough_mode: str = "direct"
    passthrough_gain: float = 1.0
    remote_to_local: DialogueDirectionConfig = field(
        default_factory=lambda: DialogueDirectionConfig(
            asr_language="en",
            translation_target="zh-TW",
            tts_voice="none",
            output_policy="direct_passthrough",
        )
    )
    local_to_remote: DialogueDirectionConfig = field(
        default_factory=lambda: DialogueDirectionConfig(
            asr_language="zh-TW",
            translation_target="en",
            tts_voice="none",
            output_policy="direct_passthrough",
        )
    )


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
    backend: str = "silero_vad"
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 520
    max_speech_duration_s: float = 10.0
    speech_pad_ms: int = 450
    rms_threshold: float = 0.02
    neural_threshold: float = 0.5
    startup_suppress_ms: int = 0


@dataclass(slots=True)
class AsrStreamingSettings:
    partial_interval_ms: int = 800
    partial_history_seconds: int = 2
    final_history_seconds: int = 12
    soft_final_audio_ms: int = 4200


@dataclass(slots=True)
class AsrConfig:
    engine: str = "faster_whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    final_beam_size: int = 3
    condition_on_previous_text: bool = True
    final_condition_on_previous_text: bool = False
    initial_prompt: str = ""
    hotwords: str = ""
    speculative_draft_model: str = ""
    speculative_num_beams: int = 1
    temperature_fallback: str = "0.0,0.2,0.4"
    no_speech_threshold: float = 0.55
    hallucination_filter: bool = True
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
    dialogue_fast: TranslationProfileConfig = field(
        default_factory=lambda: TranslationProfileConfig(
            name="dialogue_fast",
            prompt_style="literal",
            context_items=2,
            partial_trigger_tokens=10,
            max_tokens=128,
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
class LlmRuntimeConfig:
    model_path: str = r".\runtimes\models\llm\hy-mt1.5-7b.gguf"
    ctx_size: int = 4096
    gpu_layers: int = 35
    threads: int = 8
    batch_size: int = 512


@dataclass(slots=True)
class LlmConfig:
    backend: str = "local_llama_inprocess"
    model: str = DEFAULT_FIXED_LLM_MODEL
    runtime: LlmRuntimeConfig = field(default_factory=LlmRuntimeConfig)
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
    asr_pipeline: str = "v2"
    asr_v2_backend: str = "faster_whisper_v2"
    asr_v2_endpointing: str = "neural_endpoint"
    sample_rate: int = 48000
    chunk_ms: int = ASR_DEFAULT_CHUNK_MS
    passthrough_gain: float = 1.0
    tts_gain: float = 1.0
    adaptive_asr_enabled: bool = True
    adaptive_llm_enabled: bool = True
    asr_pre_roll_ms: int = 500
    latency_mode: str = "balanced"
    display_partial_strategy: str = "stable_only"
    stable_partial_min_repeats: int = 3
    partial_stability_max_delta_chars: int = 6
    asr_partial_min_audio_ms: int = 360
    asr_partial_interval_floor_ms: int = 280
    llm_partial_interval_floor_ms: int = 320
    early_final_enabled: bool = False
    tts_accept_stable_partial: bool = False
    tts_partial_min_chars: int = TTS_DEFAULT_PARTIAL_MIN_CHARS
    tts_use_speech_profile: bool = False
    asr_queue_maxsize_local: int = 256
    asr_queue_maxsize_remote: int = 256
    llm_queue_maxsize_local: int = 32
    llm_queue_maxsize_remote: int = 32
    tts_queue_maxsize_local: int = TTS_DEFAULT_QUEUE_MAXSIZE
    tts_queue_maxsize_remote: int = TTS_DEFAULT_QUEUE_MAXSIZE
    # Legacy shared queue settings kept for backward compatibility.
    asr_queue_maxsize: int = 256
    llm_queue_maxsize: int = 32
    tts_queue_maxsize: int = TTS_DEFAULT_QUEUE_MAXSIZE
    translation_exact_cache_size: int = 256
    translation_prefix_min_delta_chars: int = 6
    tts_cancel_pending_on_new_final: bool = True
    tts_cancel_policy: str = "older_only"
    tts_max_wait_ms: int = TTS_DEFAULT_MAX_WAIT_MS
    tts_max_chars: int = TTS_DEFAULT_MAX_CHARS
    tts_drop_backlog_threshold: int = TTS_DEFAULT_DROP_BACKLOG_THRESHOLD
    llm_streaming_tokens: int = 16
    max_pipeline_latency_ms: int = 3000
    asr_frontend_enabled: bool = True
    asr_frontend_target_rms: float = 0.05
    asr_frontend_max_gain: float = 3.0
    asr_frontend_highpass_alpha: float = ASR_DEFAULT_FRONTEND_HIGHPASS_ALPHA
    asr_enhancement_enabled: bool = True
    asr_enhancement_noise_reduce_strength: float = 0.42
    asr_enhancement_noise_adapt_rate: float = 0.18
    asr_enhancement_music_suppress_strength: float = 0.2
    asr_hotwords_enabled: bool = False
    asr_hotword_lexicon: str = ""
    asr_result_validator_enabled: bool = True
    asr_result_max_chars_per_second: float = 14.0
    asr_result_min_speech_ratio_for_long_text: float = 0.18
    asr_final_correction_enabled: bool = False
    asr_final_correction_context_items: int = 3
    asr_final_correction_max_chars: int = 120
    speaker_diarization_enabled: bool = False
    speaker_diarization_min_audio_ms: int = 1200
    speaker_diarization_max_speakers: int = 2
    speaker_diarization_similarity_threshold: float = 0.88
    local_echo_guard_enabled: bool = False
    local_echo_guard_resume_delay_ms: int = 300
    remote_echo_guard_resume_delay_ms: int = 300
    config_schema_version: int = 7
    session_mode: str = "meeting"
    language_config_version: int = 0
    route_config_version: int = 0
    last_migration_note: str = ""
    warmup_on_start: bool = True
    remote_translation_enabled: bool = True
    local_translation_enabled: bool = True
    translation_enabled: bool = True
    tts_output_mode: str = "subtitle_only"
    asr_language_mode: str = "fixed"
    # 新增控制項：手動固定 ASR 來源語言（auto / zh-TW / en / ja / ko / th）
    remote_asr_language: str = "en"
    local_asr_language: str = "zh-TW"
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
    # --- Phase 1: PostProcessor / Glossary ---
    glossary_enabled: bool = False
    glossary_path: str = ""
    glossary_apply_on_partial: bool = False
    glossary_apply_on_final: bool = True
    enable_postprocessor: bool = True
    enable_partial_stabilization: bool = True
    # --- Phase 1: Structured logging ---
    enable_structured_logging: bool = True
    runtime_log_format: str = "jsonl"
    # --- Phase 2: ASR profiles ---
    asr_profile_local: str = "meeting_room"
    asr_profile_remote: str = "meeting_room"
    # --- Phase 2: Degradation policy ---
    degradation_policy_enabled: bool = True
    # --- Phase 2: Final priority load control ---
    asr_final_priority_enabled: bool = True
    asr_final_priority_queue_ratio: float = 0.45
    asr_final_priority_latency_ms: int = 1800
    asr_final_priority_recover_queue_ratio: float = 0.15
    asr_final_priority_recover_after_ms: int = 8000
    # --- Phase 3: Display punctuation ---
    asr_display_punctuation_enabled: bool = False
    asr_display_punctuation_languages: str = "zh"
    asr_display_punctuation_mode: str = "lightweight"
    # --- ASR v3: confidence-gated final rescue ---
    asr_accuracy_mode: str = "balanced"
    asr_final_rescue_enabled: bool = True
    asr_final_rescue_max_attempts: int = 1
    asr_final_rescue_min_avg_logprob: float = -1.0
    asr_final_rescue_max_no_speech_prob: float = 0.65
    asr_final_rescue_max_compression_ratio: float = 2.4
    asr_final_rescue_min_chars: int = 4
    asr_chinese_fallback_enabled: bool = True
    asr_chinese_fallback_model: str = r".\runtimes\models\belle-zh-ct2"


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
    windows_io: WindowsIoConfig = field(default_factory=WindowsIoConfig)
    meeting: MeetingModeConfig = field(default_factory=MeetingModeConfig)
    dialogue: DialogueModeConfig = field(default_factory=DialogueModeConfig)
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

    @staticmethod
    def _parse_asr_config(raw: dict, fallback: "AsrConfig") -> "AsrConfig":
        """Build an AsrConfig from *raw* dict, falling back to *fallback* for missing keys."""
        return AsrConfig(
            engine=str(raw.get("engine", fallback.engine)),
            model=str(raw.get("model", fallback.model)),
            device=str(raw.get("device", fallback.device)),
            compute_type=str(raw.get("compute_type", fallback.compute_type)),
            beam_size=int(raw.get("beam_size", fallback.beam_size)),
            final_beam_size=int(raw.get("final_beam_size", fallback.final_beam_size)),
            condition_on_previous_text=bool(raw.get("condition_on_previous_text", fallback.condition_on_previous_text)),
            final_condition_on_previous_text=bool(
                raw.get("final_condition_on_previous_text", fallback.final_condition_on_previous_text)
            ),
            initial_prompt=str(raw.get("initial_prompt", fallback.initial_prompt)),
            hotwords=str(raw.get("hotwords", fallback.hotwords)),
            speculative_draft_model=str(raw.get("speculative_draft_model", fallback.speculative_draft_model)),
            speculative_num_beams=int(raw.get("speculative_num_beams", fallback.speculative_num_beams)),
            temperature_fallback=str(raw.get("temperature_fallback", fallback.temperature_fallback)),
            no_speech_threshold=float(raw.get("no_speech_threshold", fallback.no_speech_threshold)),
            hallucination_filter=bool(raw.get("hallucination_filter", fallback.hallucination_filter)),
            vad=VadSettings(**(raw.get("vad") or asdict(fallback.vad))),
            streaming=AsrStreamingSettings(**(raw.get("streaming") or asdict(fallback.streaming))),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        from app.infra.config._schema_parser import parse_app_config  # lazy – avoids circular import
        return parse_app_config(raw)

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
