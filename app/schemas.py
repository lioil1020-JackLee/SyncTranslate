from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AudioRouteConfig:
    remote_in: str = ""
    local_mic_in: str = ""
    local_tts_out: str = ""
    meeting_tts_out: str = ""


@dataclass(slots=True)
class LanguageConfig:
    remote_source: str = "en"
    remote_target: str = "zh-TW"
    local_source: str = "zh-TW"
    local_target: str = "en"


@dataclass(slots=True)
class ModelConfig:
    asr_provider: str = "mock"
    translate_provider: str = "mock"
    tts_provider: str = "mock"


@dataclass(slots=True)
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    asr_model: str = "gpt-4o-transcribe"
    translate_model: str = "gpt-5-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"


@dataclass(slots=True)
class ProviderTestStateConfig:
    asr: str = ""
    translate: str = ""
    tts: str = ""


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


@dataclass(slots=True)
class AppConfig:
    audio: AudioRouteConfig = field(default_factory=AudioRouteConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    provider_test_last_success: ProviderTestStateConfig = field(default_factory=ProviderTestStateConfig)
    session_mode: str = "remote_only"
    translate_mode: str = "fast"
    sample_rate: int = 48000
    chunk_ms: int = 100

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        audio = AudioRouteConfig(**raw.get("audio", {}))
        language = LanguageConfig(**raw.get("language", {}))
        model = ModelConfig(**raw.get("model", {}))
        openai = OpenAIConfig(**raw.get("openai", {}))
        provider_test_last_success = ProviderTestStateConfig(**raw.get("provider_test_last_success", {}))
        session_mode = str(raw.get("session_mode", "remote_only"))
        if session_mode not in {"remote_only", "local_only", "bidirectional"}:
            session_mode = "remote_only"
        translate_mode = str(raw.get("translate_mode", "fast"))
        if translate_mode not in {"fast", "quality"}:
            translate_mode = "fast"
        return cls(
            audio=audio,
            language=language,
            model=model,
            openai=openai,
            provider_test_last_success=provider_test_last_success,
            session_mode=session_mode,
            translate_mode=translate_mode,
            sample_rate=int(raw.get("sample_rate", 48000)),
            chunk_ms=int(raw.get("chunk_ms", 100)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
