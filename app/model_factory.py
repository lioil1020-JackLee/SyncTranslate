from __future__ import annotations

from app.model_providers import (
    AsrProvider,
    EdgeTtsProvider,
    LocalAsrProvider,
    MockAsrProvider,
    MockTranslateProvider,
    MockTtsProvider,
    OpenAIAsrProvider,
    OpenAITranslateProvider,
    OpenAITtsProvider,
    TranslateProvider,
    TtsProvider,
)

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "groq", "huggingface"}


def create_asr_provider(
    provider_name: str,
    language: str,
    *,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-4o-transcribe",
) -> AsrProvider:
    normalized = provider_name.strip().lower()
    if normalized == "mock":
        return MockAsrProvider(language=language)
    if normalized in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAIAsrProvider(
            language=language,
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
            model=openai_model,
        )
    if normalized == "local":
        return LocalAsrProvider(language=language, model=openai_model)
    raise ValueError(f"Unsupported ASR provider: {provider_name}")


def create_translate_provider(
    provider_name: str,
    source_lang: str,
    target_lang: str,
    *,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-5-mini",
) -> TranslateProvider:
    normalized = provider_name.strip().lower()
    if normalized == "mock":
        return MockTranslateProvider(source_lang=source_lang, target_lang=target_lang)
    if normalized in OPENAI_COMPATIBLE_PROVIDERS:
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
    normalized = provider_name.strip().lower()
    if normalized == "mock":
        return MockTtsProvider()
    if normalized in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAITtsProvider(
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
            model=openai_model,
            voice=openai_voice,
        )
    if normalized == "edge_tts":
        return EdgeTtsProvider(voice=openai_voice)
    raise ValueError(f"Unsupported TTS provider: {provider_name}")
