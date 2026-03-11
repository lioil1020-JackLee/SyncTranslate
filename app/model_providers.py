from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from typing import Protocol
from urllib import request
import wave

import numpy as np


class AsrProvider(Protocol):
    def partial_text(self) -> str: ...

    def final_text(self, audio: np.ndarray, sample_rate: int, segment_index: int) -> str: ...


class TranslateProvider(Protocol):
    def translate(self, text: str) -> str: ...


class TtsProvider(Protocol):
    def synthesize(self, text: str, sample_rate: int = 24000) -> np.ndarray: ...


@dataclass(slots=True)
class MockAsrProvider:
    language: str

    def partial_text(self) -> str:
        return f"[mock-asr {self.language}] listening..."

    def final_text(self, audio: np.ndarray, sample_rate: int, segment_index: int) -> str:
        return f"[mock-asr {self.language}] speech segment {segment_index}"


@dataclass(slots=True)
class MockTranslateProvider:
    source_lang: str
    target_lang: str

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        return f"[{self.source_lang}->{self.target_lang}] {text}"


class MockTtsProvider:
    def synthesize(self, text: str, sample_rate: int = 24000) -> np.ndarray:
        if not text.strip():
            return np.zeros((0, 1), dtype=np.float32)

        duration = min(1.5, 0.25 + len(text) * 0.015)
        samples = int(sample_rate * duration)
        t = np.linspace(0.0, duration, samples, endpoint=False)
        wave = 0.15 * np.sin(2 * np.pi * 440.0 * t)
        return wave.reshape(-1, 1).astype(np.float32)


@dataclass(slots=True)
class OpenAITranslateProvider:
    source_lang: str
    target_lang: str
    api_key_env: str
    base_url: str
    model: str
    timeout_sec: float = 20.0

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Environment variable {self.api_key_env} is not set.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. "
                        "Return only translated text without notes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate from {self.source_lang} to {self.target_lang}:\n"
                        f"{text}"
                    ),
                },
            ],
            "temperature": 0.1,
        }

        url = self.base_url.rstrip("/") + "/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI translate response has no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_chunks = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text_chunks.append(str(chunk.get("text", "")))
            return "".join(text_chunks).strip()
        return str(content).strip()


@dataclass(slots=True)
class OpenAITtsProvider:
    api_key_env: str
    base_url: str
    model: str
    voice: str
    timeout_sec: float = 30.0

    def synthesize(self, text: str, sample_rate: int = 24000) -> np.ndarray:
        if not text.strip():
            return np.zeros((0, 1), dtype=np.float32)

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Environment variable {self.api_key_env} is not set.")

        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "format": "wav",
        }
        url = self.base_url.rstrip("/") + "/audio/speech"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            wav_bytes = resp.read()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            raise ValueError(f"Unsupported sample width from OpenAI TTS: {sampwidth}")

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)[:, :1]
        else:
            samples = samples.reshape(-1, 1)
        return samples


@dataclass(slots=True)
class OpenAIAsrProvider:
    language: str
    api_key_env: str
    base_url: str
    model: str
    timeout_sec: float = 30.0

    def partial_text(self) -> str:
        return f"[openai-asr {self.language}] listening..."

    def final_text(self, audio: np.ndarray, sample_rate: int, segment_index: int) -> str:
        if audio.size == 0:
            return ""
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Environment variable {self.api_key_env} is not set.")

        wav_bytes = _audio_to_wav_bytes(audio=audio, sample_rate=sample_rate)
        boundary = "----SyncTranslateBoundary7MA4YWxkTrZu0gW"
        body = _build_multipart_form_data(
            boundary=boundary,
            fields={"model": self.model, "language": self.language},
            file_field_name="file",
            filename=f"segment_{segment_index}.wav",
            file_bytes=wav_bytes,
            content_type="audio/wav",
        )
        url = self.base_url.rstrip("/") + "/audio/transcriptions"
        req = request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload.get("text", "")
        return str(text).strip()


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = audio
    if mono.ndim == 2 and mono.shape[1] > 1:
        mono = mono[:, :1]
    mono = mono.reshape(-1)
    int16_data = np.clip(mono, -1.0, 1.0)
    int16_data = (int16_data * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
    return buffer.getvalue()


def _build_multipart_form_data(
    *,
    boundary: str,
    fields: dict[str, str],
    file_field_name: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> bytes:
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")

    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)
