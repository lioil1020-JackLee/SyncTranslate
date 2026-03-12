from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.asr_manager import ASREventWithSource
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.translation_stitcher import TranslationStitcher
from app.schemas import AppConfig


@dataclass(slots=True)
class TranslationEvent:
    source: str
    original_channel: str
    translated_channel: str
    tts_channel: str
    text: str
    is_final: bool
    should_speak: bool


class TranslatorManager:
    def __init__(self, config: AppConfig, on_error: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._on_error = on_error
        llm_cfg = config.llm
        self._llm = OllamaClient(
            backend=llm_cfg.backend,
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            request_timeout_sec=llm_cfg.request_timeout_sec,
        )
        self._stitchers = {
            "remote": TranslationStitcher(
                translator=self._llm,
                source_lang=config.language.meeting_source,
                target_lang=config.language.meeting_target,
                enabled=config.llm.sliding_window.enabled,
                trigger_tokens=config.llm.sliding_window.trigger_tokens,
                max_context_items=config.llm.sliding_window.max_context_items,
            ),
            "local": TranslationStitcher(
                translator=self._llm,
                source_lang=config.language.local_source,
                target_lang=config.language.local_target,
                enabled=config.llm.sliding_window.enabled,
                trigger_tokens=config.llm.sliding_window.trigger_tokens,
                max_context_items=config.llm.sliding_window.max_context_items,
            ),
        }

    def process(self, event: ASREventWithSource) -> TranslationEvent | None:
        source = event.source if event.source in ("local", "remote") else "local"
        stitcher = self._stitchers[source]
        try:
            stitched = stitcher.process(event)
        except Exception as exc:
            if self._on_error:
                self._on_error(f"translate_{source} failed: {exc}")
            return None
        if not stitched:
            return None

        if source == "remote":
            return TranslationEvent(
                source="remote",
                original_channel="meeting_original",
                translated_channel="meeting_translated",
                tts_channel="local",
                text=stitched.text,
                is_final=stitched.is_final,
                should_speak=stitched.should_speak,
            )
        return TranslationEvent(
            source="local",
            original_channel="local_original",
            translated_channel="local_translated",
            tts_channel="remote",
            text=stitched.text,
            is_final=stitched.is_final,
            should_speak=stitched.should_speak,
        )

    @staticmethod
    def original_channel_of(source: str) -> str:
        return "meeting_original" if source == "remote" else "local_original"
