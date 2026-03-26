from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.infra.translation.lm_studio_adapter import LmStudioClient
from app.infra.config.schema import LlmConfig, TranslationProfileConfig


@dataclass(slots=True)
class ProviderCapabilities:
    supports_json_schema: bool
    supports_chat_messages: bool
    supports_temperature: bool
    supports_keep_alive: bool
    supports_prefill: bool
    supports_response_format: bool


class TranslationProvider(Protocol):
    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
        profile: TranslationProfileConfig | None = None,
    ) -> str: ...

    def health_check(self) -> tuple[bool, str]: ...

    def list_models(self) -> list[str]: ...

    def capabilities(self) -> ProviderCapabilities: ...


class LmStudioTranslationProvider:
    def __init__(self, config: LlmConfig) -> None:
        self._client = LmStudioClient(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            repeat_penalty=config.repeat_penalty,
            stop_tokens=config.stop_tokens,
            request_timeout_sec=config.request_timeout_sec,
        )

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
        profile: TranslationProfileConfig | None = None,
    ) -> str:
        return self._client.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            profile=profile,
        )

    def health_check(self) -> tuple[bool, str]:
        return self._client.health_check()

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_json_schema=True,
            supports_chat_messages=True,
            supports_temperature=True,
            supports_keep_alive=False,
            supports_prefill=False,
            supports_response_format=True,
        )


def create_translation_provider(config: LlmConfig) -> TranslationProvider:
    backend = (config.backend or "").strip().lower()
    if backend == "lm_studio":
        return LmStudioTranslationProvider(config)
    raise ValueError(f"Unsupported llm backend: {config.backend}")
