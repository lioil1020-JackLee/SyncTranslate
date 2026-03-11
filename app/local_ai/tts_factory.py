from __future__ import annotations

from app.local_ai.piper_tts import PiperTtsEngine
from app.model_providers import EdgeTtsProvider
from app.schemas import TtsConfig


def create_tts_engine(config: TtsConfig):
    engine = (config.engine or "piper").strip().lower()
    if engine == "edge_tts":
        voice_name = config.voice_name.strip() or "zh-TW-HsiaoChenNeural"
        return EdgeTtsProvider(voice=voice_name)
    return PiperTtsEngine(
        executable_path=config.executable_path,
        model_path=config.model_path,
        config_path=config.config_path,
        speaker_id=config.speaker_id,
        length_scale=config.length_scale,
        noise_scale=config.noise_scale,
        noise_w=config.noise_w,
        sample_rate=config.sample_rate,
    )
