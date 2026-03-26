from __future__ import annotations

import unittest

from app.infra.config.schema import TtsConfig
from app.infra.tts.engine import create_tts_engine, edge_tts_rate_for_style


class TtsStylePolicyTests(unittest.TestCase):
    def test_edge_tts_style_presets_map_to_expected_rates(self) -> None:
        self.assertEqual(edge_tts_rate_for_style("balanced"), "+0%")
        self.assertEqual(edge_tts_rate_for_style("broadcast_clear"), "-8%")
        self.assertEqual(edge_tts_rate_for_style("conversational"), "+6%")
        self.assertEqual(edge_tts_rate_for_style("fast_response"), "+14%")

    def test_unknown_style_preset_falls_back_to_balanced(self) -> None:
        self.assertEqual(edge_tts_rate_for_style("unknown"), "+0%")

    def test_create_tts_engine_applies_style_preset_to_provider_rate(self) -> None:
        engine = create_tts_engine(
            TtsConfig(
                voice_name="en-US-JennyNeural",
                style_preset="fast_response",
            )
        )

        self.assertEqual(engine.voice, "en-US-JennyNeural")
        self.assertEqual(engine.rate, "+14%")


if __name__ == "__main__":
    unittest.main()
