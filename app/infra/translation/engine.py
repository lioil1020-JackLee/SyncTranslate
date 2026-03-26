from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import time
from typing import Callable

from app.domain.events import ErrorEvent
from app.infra.asr.streaming_pipeline import ASREventWithSource
from app.infra.translation.provider import create_translation_provider
from app.infra.translation.stitcher import TranslationStitcher
from app.infra.config.schema import AppConfig, LlmConfig, TranslationProfileConfig, translation_enabled_for_source


@dataclass(slots=True)
class TranslationEvent:
    source: str
    utterance_id: str
    revision: int
    created_at: float
    original_channel: str
    translated_channel: str
    tts_channel: str
    text: str
    speak_text: str
    is_final: bool
    should_speak: bool


class TranslatorManager:
    def __init__(self, config: AppConfig, on_error: Callable[[str | ErrorEvent], None] | None = None) -> None:
        self._config = config
        self._on_error = on_error
        local_llm = self._effective_llm_config(config.llm, config.llm_channels.local)
        remote_llm = self._effective_llm_config(config.llm, config.llm_channels.remote)
        self._providers = {
            "local": create_translation_provider(local_llm),
            "remote": create_translation_provider(remote_llm),
        }
        local_caption_profile = self._resolve_profile(local_llm, local_llm.caption_profile)
        remote_caption_profile = self._resolve_profile(remote_llm, remote_llm.caption_profile)
        runtime_streaming_tokens = max(8, int(getattr(config.runtime, "llm_streaming_tokens", 16)))
        local_trigger_tokens = max(
            8,
            int(local_caption_profile.partial_trigger_tokens or local_llm.sliding_window.trigger_tokens),
        )
        remote_trigger_tokens = max(
            8,
            int(remote_caption_profile.partial_trigger_tokens or remote_llm.sliding_window.trigger_tokens),
        )
        local_trigger_tokens = min(local_trigger_tokens, runtime_streaming_tokens)
        remote_trigger_tokens = min(remote_trigger_tokens, runtime_streaming_tokens)
        local_context_items = max(2, int(local_caption_profile.context_items or local_llm.sliding_window.max_context_items))
        remote_context_items = max(2, int(remote_caption_profile.context_items or remote_llm.sliding_window.max_context_items))
        self._stitchers = {
            "remote": TranslationStitcher(
                translator=self._providers["remote"],
                source_lang=config.language.meeting_source,
                target_lang=config.language.meeting_target,
                profile=remote_caption_profile,
                enabled=remote_llm.sliding_window.enabled,
                trigger_tokens=remote_trigger_tokens,
                max_context_items=remote_context_items,
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
            ),
            "local": TranslationStitcher(
                translator=self._providers["local"],
                source_lang=config.language.local_source,
                target_lang=config.language.local_target,
                profile=local_caption_profile,
                enabled=local_llm.sliding_window.enabled,
                trigger_tokens=local_trigger_tokens,
                max_context_items=local_context_items,
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
            ),
        }
        self._speech_profiles = {
            "local": self._resolve_profile(local_llm, local_llm.speech_profile),
            "remote": self._resolve_profile(remote_llm, remote_llm.speech_profile),
        }
        self._caption_profile_name = {
            "local": local_llm.caption_profile,
            "remote": remote_llm.caption_profile,
        }
        self._speech_profile_name = {
            "local": local_llm.speech_profile,
            "remote": remote_llm.speech_profile,
        }

    def translation_enabled(self, source: str | None = None) -> bool:
        if source in {"local", "remote"}:
            return translation_enabled_for_source(self._config.runtime, source)
        return any(
            translation_enabled_for_source(self._config.runtime, key)
            for key in ("local", "remote")
        )

    def process(self, event: ASREventWithSource) -> TranslationEvent | None:
        source = event.source if event.source in ("local", "remote") else "local"
        stitcher = self._stitchers[source]
        source_lang, target_lang = self._resolve_languages(source, getattr(event, "detected_language", ""))
        stitcher.set_languages(source_lang=source_lang, target_lang=target_lang)
        try:
            stitched = stitcher.process(event)
        except Exception as exc:
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="error",
                        module="translator_manager",
                        source=source,
                        code="translate_failed",
                        message="Translation failed",
                        detail=str(exc),
                    )
                )
            return None
        if not stitched:
            return None

        if source == "remote":
            speak_text = self._resolve_speak_text(
                source="remote",
                text=event.text,
                caption_text=stitched.text,
                source_lang=source_lang,
                target_lang=target_lang,
                is_final=stitched.is_final,
                should_speak=stitched.should_speak,
            )
            return TranslationEvent(
                source="remote",
                utterance_id=event.utterance_id,
                revision=event.revision,
                created_at=time.time(),
                original_channel="meeting_original",
                translated_channel="meeting_translated",
                tts_channel="local",
                text=stitched.text,
                speak_text=speak_text,
                is_final=stitched.is_final,
                should_speak=stitched.should_speak,
            )
        speak_text = self._resolve_speak_text(
            source="local",
            text=event.text,
            caption_text=stitched.text,
            source_lang=source_lang,
            target_lang=target_lang,
            is_final=stitched.is_final,
            should_speak=stitched.should_speak,
        )
        return TranslationEvent(
            source="local",
            utterance_id=event.utterance_id,
            revision=event.revision,
            created_at=time.time(),
            original_channel="local_original",
            translated_channel="local_translated",
            tts_channel="remote",
            text=stitched.text,
            speak_text=speak_text,
            is_final=stitched.is_final,
            should_speak=stitched.should_speak,
        )

    @staticmethod
    def original_channel_of(source: str) -> str:
        return "meeting_original" if source == "remote" else "local_original"

    @staticmethod
    def _resolve_profile(llm_config: LlmConfig, name: str) -> TranslationProfileConfig:
        profiles = llm_config.profiles
        mapping: dict[str, TranslationProfileConfig] = {
            "live_caption_fast": profiles.live_caption_fast,
            "live_caption_stable": profiles.live_caption_stable,
            "speech_output_natural": profiles.speech_output_natural,
            "technical_meeting": profiles.technical_meeting,
        }
        return mapping.get(name, profiles.live_caption_fast)

    def _resolve_speak_text(
        self,
        *,
        source: str,
        text: str,
        caption_text: str,
        source_lang: str,
        target_lang: str,
        is_final: bool,
        should_speak: bool,
    ) -> str:
        if not is_final or not should_speak:
            return caption_text
        if self._speech_profile_name.get(source) == self._caption_profile_name.get(source):
            return caption_text
        try:
            provider = self._providers.get(source, self._providers["local"])
            speech_profile = self._speech_profiles.get(source, self._speech_profiles["local"])
            spoken = provider.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=None,
                profile=speech_profile,
            )
            return spoken or caption_text
        except Exception as exc:
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="warning",
                        module="translator_manager",
                        code="speech_profile_fallback",
                        message="Speech profile translation failed, fallback to caption text",
                        detail=str(exc),
                    )
                )
            return caption_text

    def _resolve_languages(self, source: str, detected_language: str) -> tuple[str, str]:
        if source == "remote":
            default_source = self._config.language.meeting_source
            target = self._config.language.meeting_target
        else:
            default_source = self._config.language.local_source
            target = self._config.language.local_target

        detected = (detected_language or "").strip()
        if detected:
            return detected, target
        return default_source, target

    @staticmethod
    def _effective_llm_config(base: LlmConfig, channel: LlmConfig) -> LlmConfig:
        default = LlmConfig()
        merged = deepcopy(channel)
        scalar_fields = (
            "backend",
            "base_url",
            "model",
            "temperature",
            "top_p",
            "max_output_tokens",
            "repeat_penalty",
            "stop_tokens",
            "request_timeout_sec",
            "caption_profile",
            "speech_profile",
        )
        for field_name in scalar_fields:
            channel_value = getattr(channel, field_name)
            default_value = getattr(default, field_name)
            base_value = getattr(base, field_name)
            if channel_value == default_value and base_value != default_value:
                setattr(merged, field_name, deepcopy(base_value))

        if channel.sliding_window == default.sliding_window and base.sliding_window != default.sliding_window:
            merged.sliding_window = deepcopy(base.sliding_window)
        if channel.profiles == default.profiles and base.profiles != default.profiles:
            merged.profiles = deepcopy(base.profiles)
        return merged
