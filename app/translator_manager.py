from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from app.asr_manager import ASREventWithSource
from app.events import ErrorEvent
from app.local_ai.llm_provider import create_translation_provider
from app.local_ai.translation_stitcher import TranslationStitcher
from app.schemas import AppConfig, LlmConfig, TranslationProfileConfig


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
        local_llm = self._llm_for_direction(config.language.local_source, config.language.local_target)
        remote_llm = self._llm_for_direction(config.language.meeting_source, config.language.meeting_target)
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

    def _llm_for_direction(self, source_lang: str, target_lang: str) -> LlmConfig:
        source = (source_lang or "").strip().lower()
        target = (target_lang or "").strip().lower()
        if "-" in source:
            source = source.split("-", 1)[0]
        if "-" in target:
            target = target.split("-", 1)[0]
        # local profile = 中翻英, remote profile = 英翻中
        if source.startswith("zh") and target.startswith("en"):
            return self._config.llm_channels.local
        if source.startswith("en") and target.startswith("zh"):
            return self._config.llm_channels.remote
        return self._config.llm_channels.local

    def process(self, event: ASREventWithSource) -> TranslationEvent | None:
        source = event.source if event.source in ("local", "remote") else "local"
        stitcher = self._stitchers[source]
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
                source_lang=self._config.language.meeting_source,
                target_lang=self._config.language.meeting_target,
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
            source_lang=self._config.language.local_source,
            target_lang=self._config.language.local_target,
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
