from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from app.infra.config.schema import AsrConfig


@dataclass(slots=True)
class FrontendTuning:
    noise_reduce_strength: float
    music_suppress_strength: float


@dataclass(slots=True)
class LanguageAsrProfile:
    language: str
    endpoint_profile: str
    frontend: FrontendTuning
    asr: AsrConfig


def normalize_asr_language(language: str) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    if not value or value == "auto":
        return "auto"
    if value == "zh":
        return "zh"
    if value.startswith("zh-") or value.startswith("cmn") or value.startswith("yue"):
        return "zh"
    if value.startswith("en"):
        return "en"
    if value.startswith("ja"):
        return "ja"
    if value.startswith("ko"):
        return "ko"
    if value.startswith("th"):
        return "th"
    return value.split("-", 1)[0]


def resolve_language_asr_profile(base: AsrConfig, *, language: str) -> LanguageAsrProfile:
    normalized = normalize_asr_language(language)
    builder = _PROFILE_BUILDERS.get(normalized, _build_other_profile)
    return builder(base)


def _clone(base: AsrConfig) -> AsrConfig:
    return deepcopy(base)


def _build_zh_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 2
    profile.final_beam_size = max(4, int(profile.final_beam_size))
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.25
    if not str(profile.initial_prompt or "").strip():
        profile.initial_prompt = "以下是繁體中文口語敘事逐字稿。"
    profile.vad.min_speech_duration_ms = 200
    profile.vad.min_silence_duration_ms = 520
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 320
    profile.vad.rms_threshold = 0.022
    profile.vad.neural_threshold = 0.45
    profile.streaming.partial_interval_ms = 460
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 20
    profile.streaming.soft_final_audio_ms = 3600
    return LanguageAsrProfile(
        language="zh",
        endpoint_profile="meeting_room",
        frontend=FrontendTuning(noise_reduce_strength=0.42, music_suppress_strength=0.20),
        asr=profile,
    )


def _build_en_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 1
    profile.final_beam_size = 5
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.32
    profile.initial_prompt = ""
    profile.vad.min_speech_duration_ms = 180
    profile.vad.min_silence_duration_ms = 420
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 240
    profile.vad.rms_threshold = 0.018
    profile.vad.neural_threshold = 0.32
    profile.streaming.partial_interval_ms = 340
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 10
    profile.streaming.soft_final_audio_ms = 2200
    return LanguageAsrProfile(
        language="en",
        endpoint_profile="turn_taking",
        frontend=FrontendTuning(noise_reduce_strength=0.18, music_suppress_strength=0.0),
        asr=profile,
    )


def _build_ja_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 2
    profile.final_beam_size = 6
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.36
    profile.initial_prompt = ""
    profile.vad.min_speech_duration_ms = 180
    profile.vad.min_silence_duration_ms = 520
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 260
    profile.vad.rms_threshold = 0.018
    profile.vad.neural_threshold = 0.36
    profile.streaming.partial_interval_ms = 420
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 12
    profile.streaming.soft_final_audio_ms = 3000
    return LanguageAsrProfile(
        language="ja",
        endpoint_profile="headset",
        frontend=FrontendTuning(noise_reduce_strength=0.14, music_suppress_strength=0.0),
        asr=profile,
    )


def _build_ko_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 2
    profile.final_beam_size = 6
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.34
    profile.initial_prompt = ""
    profile.vad.min_speech_duration_ms = 180
    profile.vad.min_silence_duration_ms = 500
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 260
    profile.vad.rms_threshold = 0.019
    profile.vad.neural_threshold = 0.34
    profile.streaming.partial_interval_ms = 420
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 12
    profile.streaming.soft_final_audio_ms = 2800
    return LanguageAsrProfile(
        language="ko",
        endpoint_profile="headset",
        frontend=FrontendTuning(noise_reduce_strength=0.12, music_suppress_strength=0.0),
        asr=profile,
    )


def _build_th_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 2
    profile.final_beam_size = 6
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.28
    profile.initial_prompt = ""
    profile.vad.min_speech_duration_ms = 180
    profile.vad.min_silence_duration_ms = 560
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 280
    profile.vad.rms_threshold = 0.016
    profile.vad.neural_threshold = 0.30
    profile.streaming.partial_interval_ms = 480
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 14
    profile.streaming.soft_final_audio_ms = 3400
    return LanguageAsrProfile(
        language="th",
        endpoint_profile="meeting_room",
        frontend=FrontendTuning(noise_reduce_strength=0.08, music_suppress_strength=0.0),
        asr=profile,
    )


def _build_other_profile(base: AsrConfig) -> LanguageAsrProfile:
    profile = _clone(base)
    profile.beam_size = 2
    profile.final_beam_size = max(5, int(profile.final_beam_size))
    profile.condition_on_previous_text = False
    profile.final_condition_on_previous_text = False
    profile.temperature_fallback = "0.0"
    profile.no_speech_threshold = 0.32
    profile.initial_prompt = ""
    profile.vad.min_speech_duration_ms = 180
    profile.vad.min_silence_duration_ms = 460
    profile.vad.max_speech_duration_s = 24.0
    profile.vad.speech_pad_ms = 260
    profile.vad.rms_threshold = 0.018
    profile.vad.neural_threshold = 0.34
    profile.streaming.partial_interval_ms = 420
    profile.streaming.partial_history_seconds = 2
    profile.streaming.final_history_seconds = 12
    profile.streaming.soft_final_audio_ms = 2600
    return LanguageAsrProfile(
        language="other",
        endpoint_profile="headset",
        frontend=FrontendTuning(noise_reduce_strength=0.12, music_suppress_strength=0.0),
        asr=profile,
    )


_PROFILE_BUILDERS = {
    "zh": _build_zh_profile,
    "en": _build_en_profile,
    "ja": _build_ja_profile,
    "ko": _build_ko_profile,
    "th": _build_th_profile,
}


__all__ = [
    "FrontendTuning",
    "LanguageAsrProfile",
    "normalize_asr_language",
    "resolve_language_asr_profile",
]
