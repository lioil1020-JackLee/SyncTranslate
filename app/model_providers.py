from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import io
import json
import os
import tempfile
import time
from typing import Protocol
from urllib import error, request
import wave

import numpy as np

from app.env_vars import get_env_var


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
class LocalAsrProvider:
    language: str
    model: str = "small"
    device: str = "auto"
    compute_type: str = "int8"
    _whisper_model: object | None = field(default=None, init=False, repr=False)

    def partial_text(self) -> str:
        return f"[local-asr {self.language}] listening..."

    def final_text(self, audio: np.ndarray, sample_rate: int, segment_index: int) -> str:
        if audio.size == 0:
            return ""

        whisper_model = self._get_whisper_model()
        wav_bytes = _audio_to_wav_bytes(audio=audio, sample_rate=sample_rate)

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="synctranslate_", suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                temp_path = tmp.name

            transcribe_kwargs: dict[str, object] = {"beam_size": 1, "vad_filter": True}
            language = _normalize_whisper_language(self.language)
            if language:
                transcribe_kwargs["language"] = language

            segments, _ = whisper_model.transcribe(temp_path, **transcribe_kwargs)
            text = "".join(str(getattr(seg, "text", "")) for seg in segments).strip()
            if text:
                return text
            return f"[local-asr {self.language}] speech segment {segment_index}"
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _get_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            raise ValueError(
                "Local ASR requires optional package 'faster-whisper'. Install with: uv add faster-whisper"
            ) from exc
        self._whisper_model = WhisperModel(
            self.model or "small",
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._whisper_model


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
            # Convert legacy OpenAI voice values to a stable Edge fallback.
            voice_name = "zh-TW-HsiaoChenNeural"

        audio_bytes = _run_async_blocking(
            _edge_tts_synthesize_audio_bytes(edge_tts, text=text, voice=voice_name, rate=self.rate)
        )
        return _decode_edge_tts_audio(audio_bytes, sample_rate=sample_rate)


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

        api_key = get_env_var(self.api_key_env, "").strip()
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

        body = _openai_request_bytes(req, timeout_sec=self.timeout_sec).decode("utf-8")
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

        api_key = get_env_var(self.api_key_env, "").strip()
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
        wav_bytes = _openai_request_bytes(req, timeout_sec=self.timeout_sec)

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
        api_key = get_env_var(self.api_key_env, "").strip()
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
        payload = json.loads(_openai_request_bytes(req, timeout_sec=self.timeout_sec).decode("utf-8"))
        text = payload.get("text", "")
        return str(text).strip()


def _openai_request_bytes(req: request.Request, timeout_sec: float, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.read()
        except error.HTTPError as exc:
            body = _read_http_error_body(exc)
            retry_after = _parse_retry_after_seconds(exc.headers.get("Retry-After", "") if exc.headers else "")
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries:
                wait_sec = retry_after if retry_after is not None else min(8.0, 1.0 * (2**attempt))
                time.sleep(max(0.2, wait_sec))
                continue
            raise ValueError(_format_openai_http_error(exc.code, body, retry_after)) from exc
        except error.URLError as exc:
            if attempt < retries:
                time.sleep(min(4.0, 0.8 * (2**attempt)))
                continue
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"Network error while calling OpenAI: {reason}") from exc
    raise ValueError("OpenAI request failed after retries.")


def _read_http_error_body(exc: error.HTTPError) -> bytes:
    try:
        return exc.read()
    except Exception:
        return b""


def _parse_retry_after_seconds(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


def _format_openai_http_error(status: int, body: bytes, retry_after_sec: float | None) -> str:
    message = ""
    error_type = ""
    error_code = ""

    body_text = body.decode("utf-8", errors="replace").strip() if body else ""
    if body_text:
        try:
            payload = json.loads(body_text)
            if isinstance(payload, dict):
                error_obj = payload.get("error", {})
                if isinstance(error_obj, dict):
                    message = str(error_obj.get("message", "")).strip()
                    error_type = str(error_obj.get("type", "")).strip()
                    error_code = str(error_obj.get("code", "")).strip()
        except Exception:
            message = ""

    details: list[str] = []
    if error_code:
        details.append(error_code)
    if error_type and error_type not in details:
        details.append(error_type)
    if retry_after_sec is not None:
        details.append(f"retry_after={retry_after_sec:.1f}s")
    detail_suffix = f" ({', '.join(details)})" if details else ""

    if not message and body_text:
        message = body_text.replace("\r", " ").replace("\n", " ").strip()
    if not message:
        if status == 429:
            message = "Too Many Requests"
        else:
            message = "Request failed"
    elif len(message) > 220:
        message = message[:217] + "..."

    normalized_message = message.lower()
    if status == 403 and "error code: 1010" in normalized_message:
        message = (
            "Access denied by provider edge/CDN (Cloudflare 1010). "
            "This is usually network/IP policy, not model name format."
        )
    return f"HTTP Error {status}: {message}{detail_suffix}"


def _normalize_whisper_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if not lang:
        return ""
    if lang.startswith("zh"):
        return "zh"
    if "-" in lang:
        return lang.split("-", 1)[0]
    return lang


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
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio":
            payload = chunk.get("data")
            if isinstance(payload, bytes):
                audio_chunks.append(payload)

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
            "Install with: uv pip install miniaudio"
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
