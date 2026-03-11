from __future__ import annotations

from app.model_providers import (
    AsrProvider,
    MockAsrProvider,
    MockTranslateProvider,
    MockTtsProvider,
    OpenAIAsrProvider,
    OpenAITranslateProvider,
    OpenAITtsProvider,
    TranslateProvider,
    TtsProvider,
)


def create_asr_provider(
    provider_name: str,
    language: str,
    *,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-4o-mini-transcribe",
) -> AsrProvider:
    if provider_name == "mock":
        return MockAsrProvider(language=language)
    if provider_name == "openai":
        return OpenAIAsrProvider(
            language=language,
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
            model=openai_model,
        )
    raise ValueError(f"Unsupported ASR provider: {provider_name}")


def create_translate_provider(
    provider_name: str,
    source_lang: str,
    target_lang: str,
    *,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-4.1-mini",
) -> TranslateProvider:
    if provider_name == "mock":
        return MockTranslateProvider(source_lang=source_lang, target_lang=target_lang)
    if provider_name == "openai":
        return OpenAITranslateProvider(
            source_lang=source_lang,
            target_lang=target_lang,
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
            model=openai_model,
        )
    raise ValueError(f"Unsupported translate provider: {provider_name}")


def create_tts_provider(
    provider_name: str,
    *,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-4o-mini-tts",
    openai_voice: str = "alloy",
) -> TtsProvider:
    if provider_name == "mock":
        return MockTtsProvider()
    if provider_name == "openai":
        return OpenAITtsProvider(
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
            model=openai_model,
            voice=openai_voice,
        )
    raise ValueError(f"Unsupported TTS provider: {provider_name}")
