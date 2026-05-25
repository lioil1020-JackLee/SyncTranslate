from __future__ import annotations

import numpy as np

from app.infra.asr.resampling import resample_audio
from app.infra.audio.format_policy import ASR_SAMPLE_RATE, WINDOWS_AUDIO_SAMPLE_RATE
from app.infra.audio.frame import AudioFrame


def ensure_float32_frame(
    audio: np.ndarray,
    sample_rate: int,
    source_id: str,
    role: str,
    *,
    source_type: str = "input",
) -> AudioFrame:
    return AudioFrame.from_samples(
        audio,
        sample_rate=int(sample_rate),
        source_id=source_id,
        role=role,
        source_type=source_type,
    )


def to_asr_mono_16k(frame: AudioFrame) -> np.ndarray:
    samples = np.asarray(frame.samples, dtype=np.float32)
    if samples.ndim == 1:
        mono = samples
    elif samples.shape[1] == 1:
        mono = samples[:, 0]
    else:
        mono = np.mean(samples[:, : min(samples.shape[1], 2)], axis=1, dtype=np.float32)
    return resample_audio(mono, sample_rate=int(frame.sample_rate), target_rate=ASR_SAMPLE_RATE).astype(np.float32, copy=False)


def soft_limiter(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= threshold:
        return y.astype(np.float32, copy=False)
    return (np.tanh(y / float(threshold)) * float(threshold)).astype(np.float32, copy=False)


def to_output_float32_stereo_48k(frame: AudioFrame, *, gain: float = 1.0) -> np.ndarray:
    samples = np.asarray(frame.samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape((-1, 1))
    if samples.shape[1] == 1:
        stereo = np.repeat(samples, 2, axis=1)
    elif samples.shape[1] == 2:
        stereo = samples
    else:
        stereo = samples[:, :2]
    if int(frame.sample_rate) != WINDOWS_AUDIO_SAMPLE_RATE:
        left = resample_audio(stereo[:, 0], sample_rate=int(frame.sample_rate), target_rate=WINDOWS_AUDIO_SAMPLE_RATE)
        right = resample_audio(stereo[:, 1], sample_rate=int(frame.sample_rate), target_rate=WINDOWS_AUDIO_SAMPLE_RATE)
        size = min(left.shape[0], right.shape[0])
        stereo = np.column_stack((left[:size], right[:size])).astype(np.float32, copy=False)
    if gain != 1.0:
        stereo = stereo.astype(np.float32, copy=True) * max(0.0, float(gain))
    return soft_limiter(stereo)


def to_pcm16_stereo_48k(
    frame_or_audio: AudioFrame | np.ndarray,
    sample_rate: int | None = None,
    *,
    gain: float = 1.0,
) -> bytes:
    if isinstance(frame_or_audio, AudioFrame):
        stereo = to_output_float32_stereo_48k(frame_or_audio, gain=gain)
    else:
        if sample_rate is None:
            sample_rate = WINDOWS_AUDIO_SAMPLE_RATE
        frame = ensure_float32_frame(frame_or_audio, int(sample_rate), "", "boundary")
        stereo = to_output_float32_stereo_48k(frame, gain=gain)
    pcm = np.clip(stereo, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2", copy=False).tobytes()


__all__ = [
    "ensure_float32_frame",
    "soft_limiter",
    "to_asr_mono_16k",
    "to_output_float32_stereo_48k",
    "to_pcm16_stereo_48k",
]
