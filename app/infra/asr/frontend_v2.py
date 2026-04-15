from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.infra.asr.enhancement_v2 import AsrSpeechEnhancerV2


@dataclass(slots=True)
class FrontendChunk:
    audio: np.ndarray
    sample_rate: int
    input_rms: float
    output_rms: float
    applied_gain: float
    clipped_ratio: float
    speech_ratio: float
    noise_floor_rms: float
    suppression_ratio: float
    music_likelihood: float


class AsrAudioFrontendV2:
    """Stateful light-weight audio front-end for ASR v2.

    This intentionally stays inexpensive: it does not try to do full denoise or
    source separation, but it restores the practical pieces that desktop/mobile
    dictation stacks rely on before VAD/ASR:
    - stable mono collapse
    - DC offset removal
    - mild AGC for quiet speech
    - light first-order high-pass / pre-emphasis to preserve speech on music beds
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        target_rms: float = 0.05,
        max_gain: float = 3.0,
        highpass_alpha: float = 0.96,
        enhancement_enabled: bool = True,
        enhancement_noise_reduce_strength: float = 0.42,
        enhancement_noise_adapt_rate: float = 0.18,
        enhancement_music_suppress_strength: float = 0.2,
        speech_frame_ms: int = 20,
        speech_gate_ratio: float = 1.35,
    ) -> None:
        self._enabled = bool(enabled)
        self._target_rms = max(0.01, float(target_rms))
        self._max_gain = max(1.0, float(max_gain))
        self._highpass_alpha = min(0.999, max(0.0, float(highpass_alpha)))
        self._enhancer = AsrSpeechEnhancerV2(
            enabled=enhancement_enabled,
            noise_reduce_strength=enhancement_noise_reduce_strength,
            noise_adapt_rate=enhancement_noise_adapt_rate,
            music_suppress_strength=enhancement_music_suppress_strength,
        )
        self._speech_frame_ms = max(10, int(speech_frame_ms))
        self._speech_gate_ratio = max(1.0, float(speech_gate_ratio))
        self._prev_input = 0.0
        self._prev_output = 0.0
        self._last_stats = {
            "input_rms": 0.0,
            "output_rms": 0.0,
            "applied_gain": 1.0,
            "clipped_ratio": 0.0,
            "speech_ratio": 0.0,
            "noise_floor_rms": 0.0,
            "suppression_ratio": 0.0,
            "music_likelihood": 0.0,
        }

    def reset(self) -> None:
        self._prev_input = 0.0
        self._prev_output = 0.0
        self._enhancer.reset()
        self._last_stats = {
            "input_rms": 0.0,
            "output_rms": 0.0,
            "applied_gain": 1.0,
            "clipped_ratio": 0.0,
            "speech_ratio": 0.0,
            "noise_floor_rms": 0.0,
            "suppression_ratio": 0.0,
            "music_likelihood": 0.0,
        }

    def process(self, chunk: np.ndarray, sample_rate: float) -> FrontendChunk:
        sample_rate_int = int(sample_rate) if sample_rate > 0 else 16000
        mono = self._collapse_channels(chunk)
        if mono.size == 0:
            return FrontendChunk(
                audio=mono.astype(np.float32, copy=False),
                sample_rate=sample_rate_int,
                input_rms=0.0,
                output_rms=0.0,
                applied_gain=1.0,
                clipped_ratio=0.0,
                speech_ratio=0.0,
                noise_floor_rms=0.0,
                suppression_ratio=0.0,
                music_likelihood=0.0,
            )
        if not self._enabled:
            mono = mono.reshape(-1).astype(np.float32, copy=False)
            rms = _rms(mono)
            self._last_stats = {
                "input_rms": round(rms, 5),
                "output_rms": round(rms, 5),
                "applied_gain": 1.0,
                "clipped_ratio": 0.0,
                "speech_ratio": 0.0,
                "noise_floor_rms": 0.0,
                "suppression_ratio": 0.0,
                "music_likelihood": 0.0,
            }
            return FrontendChunk(
                audio=mono,
                sample_rate=sample_rate_int,
                input_rms=rms,
                output_rms=rms,
                applied_gain=1.0,
                clipped_ratio=0.0,
                speech_ratio=0.0,
                noise_floor_rms=0.0,
                suppression_ratio=0.0,
                music_likelihood=0.0,
            )

        mono = mono.astype(np.float32, copy=False)
        input_rms = _rms(mono)
        signal_std = float(np.std(mono)) if mono.size else 0.0
        apply_highpass = signal_std >= 1e-4
        if apply_highpass:
            mono = mono - float(np.mean(mono))
        speech_ratio = self._speech_ratio(mono, sample_rate_int, baseline_rms=input_rms)
        if apply_highpass:
            mono = self._highpass(mono)
        enhanced = self._enhancer.process(mono, sample_rate_int, speech_ratio=speech_ratio)
        mono = enhanced.audio
        speech_ratio = max(speech_ratio, self._speech_ratio(mono, sample_rate_int, baseline_rms=max(_rms(mono), 1e-6)))
        gain = 1.0
        enhanced_rms = _rms(mono)
        if enhanced_rms > 1e-6:
            gain = min(self._max_gain, self._target_rms / enhanced_rms)
            # Keep music/noise pumping under control when the chunk looks weakly voiced.
            if speech_ratio < 0.25:
                gain = min(gain, 1.8)
        processed = np.clip(mono * gain, -1.0, 1.0)
        clipped_ratio = float(np.mean(np.abs(processed) >= 0.995)) if processed.size else 0.0
        output_rms = _rms(processed)
        self._last_stats = {
            "input_rms": round(input_rms, 5),
            "output_rms": round(output_rms, 5),
            "applied_gain": round(gain, 3),
            "clipped_ratio": round(clipped_ratio, 5),
            "speech_ratio": round(speech_ratio, 4),
            "noise_floor_rms": round(enhanced.noise_floor_rms, 5),
            "suppression_ratio": round(enhanced.suppression_ratio, 4),
            "music_likelihood": round(enhanced.music_likelihood, 4),
        }
        return FrontendChunk(
            audio=processed.astype(np.float32, copy=False),
            sample_rate=sample_rate_int,
            input_rms=input_rms,
            output_rms=output_rms,
            applied_gain=gain,
            clipped_ratio=clipped_ratio,
            speech_ratio=speech_ratio,
            noise_floor_rms=enhanced.noise_floor_rms,
            suppression_ratio=enhanced.suppression_ratio,
            music_likelihood=enhanced.music_likelihood,
        )

    def stats(self) -> dict[str, float]:
        return dict(self._last_stats)

    @staticmethod
    def _collapse_channels(chunk: np.ndarray) -> np.ndarray:
        payload = np.asarray(chunk, dtype=np.float32)
        if payload.ndim != 2 or payload.shape[1] <= 1:
            return payload.reshape(-1).astype(np.float32, copy=False)

        channel_energy = np.sqrt(np.mean(np.square(payload), axis=0, dtype=np.float32))
        strongest = int(np.argmax(channel_energy)) if channel_energy.size else 0
        strongest_energy = float(channel_energy[strongest]) if channel_energy.size else 0.0
        weakest_energy = float(np.min(channel_energy)) if channel_energy.size else 0.0
        if strongest_energy >= max(0.01, weakest_energy * 2.5):
            return payload[:, strongest].astype(np.float32, copy=False)

        left = payload[:, 0].astype(np.float32, copy=False)
        right = payload[:, 1].astype(np.float32, copy=False)
        left_std = float(np.std(left))
        right_std = float(np.std(right))
        if left_std > 1e-6 and right_std > 1e-6:
            corr = float(np.corrcoef(left, right)[0, 1])
            if corr >= 0.3:
                return ((left + right) * 0.5).astype(np.float32, copy=False)
            if corr <= -0.3:
                return payload[:, strongest].astype(np.float32, copy=False)
        return payload[:, strongest].astype(np.float32, copy=False)

    def _highpass(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)
        out = np.empty_like(audio, dtype=np.float32)
        prev_in = float(self._prev_input)
        prev_out = float(self._prev_output)
        alpha = self._highpass_alpha
        for idx, value in enumerate(audio):
            current = float(value)
            filtered = current - prev_in + alpha * prev_out
            out[idx] = filtered
            prev_in = current
            prev_out = filtered
        self._prev_input = prev_in
        self._prev_output = prev_out
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    def _speech_ratio(self, audio: np.ndarray, sample_rate: int, *, baseline_rms: float) -> float:
        if audio.size == 0 or sample_rate <= 0:
            return 0.0
        frame = max(64, int(sample_rate * self._speech_frame_ms / 1000.0))
        if audio.size <= frame:
            return 1.0 if baseline_rms > 0.0 else 0.0
        threshold = max(0.003, baseline_rms * self._speech_gate_ratio)
        voiced = 0
        total = 0
        for start in range(0, max(1, audio.size - frame + 1), frame):
            end = min(audio.size, start + frame)
            rms = _rms(audio[start:end])
            if rms >= threshold:
                voiced += 1
            total += 1
        return float(voiced) / float(max(1, total))


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))


__all__ = ["AsrAudioFrontendV2", "FrontendChunk"]
