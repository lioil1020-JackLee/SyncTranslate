from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from copy import deepcopy
import time
from typing import Callable

from app.infra.asr.text_correction import AsrTextCorrector
from app.domain.events import ErrorEvent
from app.infra.asr.contracts import ASREventWithSource
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
    is_stable_partial: bool
    is_early_final: bool
    should_display: bool
    should_speak: bool
    speaker_label: str = ""


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
                stable_profile=self._resolve_profile(remote_llm, "live_caption_stable"),
                enabled=remote_llm.sliding_window.enabled,
                trigger_tokens=remote_trigger_tokens,
                max_context_items=remote_context_items,
                min_partial_interval_ms=int(getattr(config.runtime, "llm_partial_interval_floor_ms", 320)),
                partial_interval_floor_ms=int(getattr(config.runtime, "llm_partial_interval_floor_ms", 320)),
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
                adaptive_enabled=bool(getattr(config.runtime, "adaptive_llm_enabled", True)),
            ),
            "local": TranslationStitcher(
                translator=self._providers["local"],
                source_lang=config.language.local_source,
                target_lang=config.language.local_target,
                profile=local_caption_profile,
                stable_profile=self._resolve_profile(local_llm, "live_caption_stable"),
                enabled=local_llm.sliding_window.enabled,
                trigger_tokens=local_trigger_tokens,
                max_context_items=local_context_items,
                min_partial_interval_ms=int(getattr(config.runtime, "llm_partial_interval_floor_ms", 320)),
                partial_interval_floor_ms=int(getattr(config.runtime, "llm_partial_interval_floor_ms", 320)),
                exact_cache_size=config.runtime.translation_exact_cache_size,
                prefix_min_delta_chars=config.runtime.translation_prefix_min_delta_chars,
                adaptive_enabled=bool(getattr(config.runtime, "adaptive_llm_enabled", True)),
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
        correction_enabled = bool(getattr(config.runtime, "asr_final_correction_enabled", False))
        self._correction_enabled = correction_enabled
        self._correction_auto_sources = {
            "local": self._supports_auto_asr_correction(local_llm),
            "remote": self._supports_auto_asr_correction(remote_llm),
        }
        correction_context_items = int(getattr(config.runtime, "asr_final_correction_context_items", 3))
        correction_max_chars = int(getattr(config.runtime, "asr_final_correction_max_chars", 120))
        self._correctors = {
            "local": AsrTextCorrector(
                local_llm,
                enabled=correction_enabled or self._correction_auto_sources["local"],
                context_items=correction_context_items,
                max_chars=correction_max_chars,
            ),
            "remote": AsrTextCorrector(
                remote_llm,
                enabled=correction_enabled or self._correction_auto_sources["remote"],
                context_items=correction_context_items,
                max_chars=correction_max_chars,
            ),
        }

    def last_skip_reason(self, source: str) -> str:
        key = source if source in {"local", "remote"} else "local"
        stitcher = self._stitchers.get(key)
        if stitcher is None:
            return ""
        return stitcher.last_skip_reason()

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
            debug_snapshot = getattr(self._providers.get(source, self._providers["local"]), "debug_snapshot", lambda: {})()
            extra = ""
            if debug_snapshot:
                extra = (
                    f" | raw={debug_snapshot.get('raw_response', '')!r}"
                    f" | cleaned={debug_snapshot.get('cleaned_response', '')!r}"
                    f" | provider_error={debug_snapshot.get('last_error', '')!r}"
                )
            if self._on_error:
                self._on_error(
                    ErrorEvent(
                        level="error",
                        module="translator_manager",
                        source=source,
                        code="translate_failed",
                        message="Translation failed",
                        detail=str(exc) + extra,
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
                is_stable_partial=stitched.is_stable_partial,
                is_early_final=stitched.is_early_final,
                should_display=stitched.should_display,
                should_speak=stitched.should_speak,
                speaker_label=event.speaker_label,
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
            is_stable_partial=stitched.is_stable_partial,
            is_early_final=stitched.is_early_final,
            should_display=stitched.should_display,
            should_speak=stitched.should_speak,
            speaker_label=event.speaker_label,
        )

    def correct_asr_event(self, event: ASREventWithSource) -> ASREventWithSource:
        if not event.is_final or bool(getattr(event, "correction_applied", False)):
            return event
        source = event.source if event.source in ("local", "remote") else "local"
        corrector = self._correctors.get(source)
        if corrector is None:
            return event
        language, _ = self._resolve_languages(source, getattr(event, "detected_language", ""))
        if not self._should_attempt_asr_correction(source=source, text=event.text, language=language):
            return event
        result = corrector.correct(event.text, language=language)
        if not result.applied:
            return event
        return ASREventWithSource(
            source=event.source,
            utterance_id=event.utterance_id,
            revision=event.revision,
            pipeline_revision=event.pipeline_revision,
            config_fingerprint=event.config_fingerprint,
            created_at=event.created_at,
            text=result.text,
            is_final=event.is_final,
            is_early_final=event.is_early_final,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            latency_ms=event.latency_ms,
            detected_language=event.detected_language,
            raw_text=event.raw_text or result.raw_text,
            correction_applied=True,
            speaker_label=event.speaker_label,
        )

    @staticmethod
    def original_channel_of(source: str) -> str:
        return "meeting_original" if source == "remote" else "local_original"

    @staticmethod
    def _resolve_profile(llm_config: LlmConfig, name: str) -> TranslationProfileConfig:
        profiles = llm_config.profiles
        mapping: dict[str, TranslationProfileConfig] = {
            "live_caption_fast": profiles.live_caption_fast,
            "dialogue_fast": profiles.dialogue_fast,
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
        return caption_text

    def _resolve_languages(self, source: str, detected_language: str) -> tuple[str, str]:
        if source == "remote":
            default_source = self._config.language.meeting_source
            target = self._config.language.meeting_target
            configured_asr_source = getattr(self._config.runtime, "remote_asr_language", "auto")
        else:
            default_source = self._config.language.local_source
            target = self._config.language.local_target
            configured_asr_source = getattr(self._config.runtime, "local_asr_language", "auto")

        detected = (detected_language or "").strip()
        if configured_asr_source and configured_asr_source != "auto":
            return configured_asr_source, target
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

    @staticmethod
    def _supports_auto_asr_correction(config: LlmConfig) -> bool:
        return str(getattr(config, "backend", "") or "").strip().lower() == "lm_studio"

    def _should_attempt_asr_correction(self, *, source: str, text: str, language: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        if self._correction_enabled:
            return True
        if not self._correction_auto_sources.get(source, False):
            return False
        normalized_language = str(language or "").strip().lower()
        compact = "".join(value.split())
        if not compact:
            return False
        cjk_count = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
        if normalized_language.startswith(("zh", "cmn", "yue")) or cjk_count >= max(2, len(compact) // 2):
            return 4 <= len(compact) <= 32 and cjk_count >= 2
        return False
