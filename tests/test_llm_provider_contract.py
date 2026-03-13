from __future__ import annotations

import unittest

from app.local_ai.llm_provider import create_translation_provider
from app.schemas import LlmConfig


class LlmProviderContractTests(unittest.TestCase):
    def test_factory_returns_provider_with_capabilities(self) -> None:
        provider = create_translation_provider(LlmConfig())
        caps = provider.capabilities()
        self.assertTrue(caps.supports_chat_messages)
        self.assertTrue(caps.supports_temperature)

    def test_unsupported_backend_raises(self) -> None:
        cfg = LlmConfig(backend="unsupported")
        with self.assertRaises(ValueError):
            create_translation_provider(cfg)


if __name__ == "__main__":
    unittest.main()
