from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
import wave

import numpy as np


@dataclass(slots=True)
class EdgeTtsProvider:
    voice: str = "zh-TW-HsiaoChenNeural"
    rate: str = "+0%"

    def synthesize(self, text: str, sample_rate: int = 24000) -> np.ndarray:
        if not text.strip():
            return np.zeros((0, 1), dtype=np.float32)

        try:
            import edge_tts  # type: ignore
        except Exception as exc:
            raise ValueError("edge_tts provider requires optional package 'edge-tts'. Install with: uv add edge-tts") from exc

        voice_name = self.voice.strip() or "zh-TW-HsiaoChenNeural"
        if "Neural" not in voice_name:
            voice_name = "zh-TW-HsiaoChenNeural"

        audio_bytes = _run_async_blocking(
            _edge_tts_synthesize_audio_bytes(edge_tts, text=text, voice=voice_name, rate=self.rate)
        )
        return _decode_edge_tts_audio(audio_bytes, sample_rate=sample_rate)


def _run_async_blocking(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


async def _edge_tts_synthesize_audio_bytes(edge_tts_module, *, text: str, voice: str, rate: str) -> bytes:
    communicator = _create_edge_communicate(edge_tts_module, text=text, voice=voice, rate=rate)
    audio_chunks: list[bytes] = []
    try:
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":
                payload = chunk.get("data")
                if isinstance(payload, bytes):
                    audio_chunks.append(payload)
    except Exception as exc:
        message = str(exc).strip()
        if "No audio was received" in message:
            raise ValueError(
                f"No audio was received from edge-tts for voice '{voice}'. "
                "This usually means the voice locale is incompatible with the input text."
            ) from exc
        raise

    payload = b"".join(audio_chunks)
    if payload:
        return payload
    raise ValueError("edge_tts returned empty audio stream.")


def _create_edge_communicate(edge_tts_module, *, text: str, voice: str, rate: str):
    kwargs = {
        "text": text,
        "voice": voice,
        "rate": rate,
    }
    for extra_kwargs in (
        {"output_format": "riff-24khz-16bit-mono-pcm"},
        {},
    ):
        try:
            merged = {**kwargs, **extra_kwargs}
            return edge_tts_module.Communicate(**merged)
        except TypeError:
            continue
    return edge_tts_module.Communicate(text, voice)


def _decode_edge_tts_audio(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    if audio_bytes.startswith(b"RIFF"):
        return _wav_bytes_to_audio(audio_bytes)
    try:
        import miniaudio  # type: ignore
    except Exception as exc:
        raise ValueError(
            "edge_tts returned MP3 audio and optional package 'miniaudio' is not installed. "
            "Sync dependencies with uv so 'miniaudio' is installed from pyproject.toml."
        ) from exc

    try:
        decoded = miniaudio.decode(
            audio_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        raise ValueError(f"edge_tts audio decode failed: {exc}") from exc

    samples = np.array(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
    return samples.reshape(-1, 1)


def _wav_bytes_to_audio(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Unsupported sample width from WAV payload: {sampwidth}")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, :1]
    else:
        samples = samples.reshape(-1, 1)
    return samples
