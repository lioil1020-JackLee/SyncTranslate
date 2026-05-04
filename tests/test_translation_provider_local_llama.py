from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.infra.config.schema import LlmConfig
from app.infra.translation.provider import LocalLlamaTranslationProvider, create_translation_provider


def test_create_translation_provider_uses_local_llama_backend() -> None:
    cfg = LlmConfig(backend="local_llama_inprocess")
    with patch("app.infra.translation.provider.InProcessLlamaClient"):
        provider = create_translation_provider(cfg)

    assert isinstance(provider, LocalLlamaTranslationProvider)


def test_local_llama_provider_delegates_model_list_to_inprocess_client() -> None:
    cfg = LlmConfig(backend="local_llama_inprocess")
    fake_client = MagicMock()
    fake_client.list_models.return_value = ["hy-mt1.5-7b"]

    with patch("app.infra.translation.provider.InProcessLlamaClient", return_value=fake_client) as ctor:
        provider = LocalLlamaTranslationProvider(cfg)
        models = provider.list_models()

    assert models == ["hy-mt1.5-7b"]
    ctor.assert_called_once()
    fake_client.list_models.assert_called_once()
