from __future__ import annotations

from app.model_providers import EdgeTtsProvider
from app.schemas import TtsConfig


def create_tts_engine(config: TtsConfig):
    voice_name = config.voice_name.strip() or "zh-TW-HsiaoChenNeural"
    return EdgeTtsProvider(voice=voice_name)
