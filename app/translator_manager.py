from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from app.asr_manager import ASREventWithSource
from app.events import ErrorEvent
from app.local_ai.llm_provider import create_translation_provider
from app.local_ai.translation_stitcher import TranslationStitcher
from app.schemas import AppConfig, TranslationProfileConfig


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
        self._provider = create_translation_provider(config.llm)
        caption_profile = self._resolve_profile(config.llm.caption_profile)
        trigger_tokens = max(8, int(caption_profile.partial_trigger_tokens or config.llm.sliding_window.trigger_tokens))
        context_items = max(2, int(caption_profile.context_items or config.llm.sliding_window.max_context_items))
        self._stitchers = {
            "remote": TranslationStitcher(
                translator=self._provider,
                source_lang=config.language.meeting_source,
                target_lang=config.language.meeting_target,
                profile=caption_profile,
                enabled=config.llm.sliding_window.enabled,
                trigger_tokens=trigger_tokens,
                max_context_items=context_items,
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
            ),
            "local": TranslationStitcher(
                translator=self._provider,
                source_lang=config.language.local_source,
                target_lang=config.language.local_target,
                profile=caption_profile,
                enabled=config.llm.sliding_window.enabled,
                trigger_tokens=trigger_tokens,
                max_context_items=context_items,
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
            ),
        }
        self._speech_profile = self._resolve_profile(config.llm.speech_profile)
        self._caption_profile_name = config.llm.caption_profile
        self._speech_profile_name = config.llm.speech_profile

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

    def _resolve_profile(self, name: str) -> TranslationProfileConfig:
        profiles = self._config.llm.profiles
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
        text: str,
        caption_text: str,
        source_lang: str,
        target_lang: str,
        is_final: bool,
        should_speak: bool,
    ) -> str:
        if not is_final or not should_speak:
            return caption_text
        if self._speech_profile_name == self._caption_profile_name:
            return caption_text
        try:
            spoken = self._provider.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=None,
                profile=self._speech_profile,
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
