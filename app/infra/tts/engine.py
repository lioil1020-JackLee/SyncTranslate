from __future__ import annotations

from app.infra.tts.edge_tts_adapter import EdgeTtsProvider
from app.infra.config.schema import TtsConfig


_EDGE_TTS_STYLE_RATES = {
    "balanced": "+0%",
    "broadcast_clear": "-8%",
    "conversational": "+6%",
    "fast_response": "+14%",
}


def edge_tts_rate_for_style(style_preset: str) -> str:
    preset = (style_preset or "").strip().lower()
    return _EDGE_TTS_STYLE_RATES.get(preset, _EDGE_TTS_STYLE_RATES["balanced"])


def create_tts_engine(config: TtsConfig):
    voice_name = config.voice_name.strip() or "zh-TW-HsiaoChenNeural"
    return EdgeTtsProvider(
        voice=voice_name,
        rate=edge_tts_rate_for_style(getattr(config, "style_preset", "balanced")),
    )
