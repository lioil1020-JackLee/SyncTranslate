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
    mode: str = "meeting_to_local"


@dataclass(slots=True)
class LanguageConfig:
    meeting_source: str = "en"
    meeting_target: str = "zh-TW"
    local_source: str = "zh-TW"
    local_target: str = "en"


@dataclass(slots=True)
class VadSettings:
    enabled: bool = True
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1500
    max_speech_duration_s: float = 15.0
    speech_pad_ms: int = 300
    rms_threshold: float = 0.01


@dataclass(slots=True)
class AsrStreamingSettings:
    partial_interval_ms: int = 400
    partial_history_seconds: int = 8
    final_history_seconds: int = 20


@dataclass(slots=True)
class AsrConfig:
    engine: str = "faster_whisper"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    condition_on_previous_text: bool = True
    vad: VadSettings = field(default_factory=VadSettings)
    streaming: AsrStreamingSettings = field(default_factory=AsrStreamingSettings)


@dataclass(slots=True)
class SlidingWindowConfig:
    enabled: bool = True
    trigger_tokens: int = 20
    max_context_items: int = 6


@dataclass(slots=True)
class LlmConfig:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    top_p: float = 0.9
    request_timeout_sec: int = 20
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)


@dataclass(slots=True)
class TtsConfig:
    engine: str = "edge_tts"
    executable_path: str = ""
    model_path: str = ""
    config_path: str = ""
    voice_name: str = "zh-TW-HsiaoChenNeural"
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
    asr_queue_maxsize: int = 128
    llm_queue_maxsize: int = 32
    tts_queue_maxsize: int = 32
    warmup_on_start: bool = True


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
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    meeting_tts: TtsConfig = field(default_factory=TtsConfig)
    local_tts: TtsConfig = field(default_factory=TtsConfig)
    tts_channels: TtsChannelsConfig = field(default_factory=TtsChannelsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    health_last_success: HealthStateConfig = field(default_factory=HealthStateConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        audio = AudioRouteConfig(**(raw.get("audio") or {}))
        direction = DirectionConfig(**(raw.get("direction") or {}))
        language = LanguageConfig(**(raw.get("language") or {}))

        asr_raw = raw.get("asr") or {}
        asr = AsrConfig(
            engine=str(asr_raw.get("engine", "faster_whisper")),
            model=str(asr_raw.get("model", "large-v3")),
            device=str(asr_raw.get("device", "cuda")),
            compute_type=str(asr_raw.get("compute_type", "float16")),
            beam_size=int(asr_raw.get("beam_size", 1)),
            condition_on_previous_text=bool(asr_raw.get("condition_on_previous_text", True)),
            vad=VadSettings(**(asr_raw.get("vad") or {})),
            streaming=AsrStreamingSettings(**(asr_raw.get("streaming") or {})),
        )

        llm_raw = raw.get("llm") or {}
        llm = LlmConfig(
            backend=str(llm_raw.get("backend", "ollama")),
            base_url=str(llm_raw.get("base_url", "http://127.0.0.1:11434")),
            model=str(llm_raw.get("model", "llama3.1:8b")),
            temperature=float(llm_raw.get("temperature", 0.2)),
            top_p=float(llm_raw.get("top_p", 0.9)),
            request_timeout_sec=int(llm_raw.get("request_timeout_sec", 20)),
            sliding_window=SlidingWindowConfig(**(llm_raw.get("sliding_window") or {})),
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

        if direction.mode not in {"meeting_to_local", "local_to_meeting", "bidirectional"}:
            direction.mode = "meeting_to_local"
        if llm.backend not in {"ollama", "lm_studio"}:
            llm.backend = "ollama"

        return cls(
            audio=audio,
            direction=direction,
            language=language,
            asr=asr,
            llm=llm,
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
        speaker_id=(override.speaker_id if override.speaker_id is not None else fallback.speaker_id),
        length_scale=(override.length_scale if override.length_scale is not None else fallback.length_scale),
        noise_scale=(override.noise_scale if override.noise_scale is not None else fallback.noise_scale),
        noise_w=(override.noise_w if override.noise_w is not None else fallback.noise_w),
        sample_rate=(override.sample_rate if override.sample_rate is not None else fallback.sample_rate),
    )
