from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AudioRouteConfig:
    meeting_in: str = ""
    microphone_in: str = ""
    speaker_out: str = ""
    meeting_out: str = ""


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
    engine: str = "piper"
    executable_path: str = ".\\tools\\piper\\piper.exe"
    model_path: str = ".\\models\\tts\\zh-TW-medium.onnx"
    config_path: str = ".\\models\\tts\\zh-TW-medium.onnx.json"
    voice_name: str = ""
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sample_rate: int = 22050


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
            runtime=runtime,
            health_last_success=health_last_success,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
