"""AudioFrontendChain — composable pipeline of AudioProcessingStages.

Replaces the monolithic AsrAudioFrontendV2 with a clean stage-based chain.
The chain is backward-compatible: it produces the same FrontendChunk output
and wraps the same underlying processing.

Usage
-----
chain = AudioFrontendChain.build_default(
    enabled=True,
    target_rms=0.05,
    max_gain=3.0,
    highpass_alpha=0.96,
    enhancement_enabled=True,
    enhancement_noise_reduce_strength=0.42,
    enhancement_noise_adapt_rate=0.18,
    enhancement_music_suppress_strength=0.2,
)
result = chain.process(audio, sample_rate)  # returns FrontendChunk

Future AEC integration point
-----------------------------
Insert a ReferenceAwareAudioStage before HighpassStage:

    chain.insert(0, MyAECStage(reference_buffer=playback_buffer))

The stage receives audio and can optionally receive a reference signal via
``process_with_reference(audio, sr, reference=ref_audio)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.infra.asr.audio_pipeline.highpass import HighpassStage
from app.infra.asr.audio_pipeline.identity import IdentityStage
from app.infra.asr.audio_pipeline.loudness import LoudnessStage
from app.infra.asr.audio_pipeline.music_suppression import MusicSuppressionStage
from app.infra.asr.audio_pipeline.noise_reduction import NoiseReductionStage


@dataclass(slots=True)
class ChainResult:
    """Output of AudioFrontendChain.process()."""
    audio: np.ndarray
    sample_rate: int
    input_rms: float
    output_rms: float
    applied_gain: float


class AudioFrontendChain:
    """Ordered list of AudioProcessingStages executed sequentially.

    Parameters
    ----------
    stages:
        Ordered list of stage objects.  Each must implement
        ``process(audio, sample_rate) -> np.ndarray`` and ``reset()``.
    enabled:
        Master switch.  When False, all stages are bypassed.
    """

    def __init__(self, stages: list[object], *, enabled: bool = True) -> None:
        self._stages = list(stages)
        self._enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        *,
        enabled: bool = True,
        target_rms: float = 0.05,
        max_gain: float = 3.0,
        highpass_alpha: float = 0.96,
        enhancement_enabled: bool = True,
        enhancement_noise_reduce_strength: float = 0.42,
        enhancement_noise_adapt_rate: float = 0.18,
        enhancement_music_suppress_strength: float = 0.2,
    ) -> "AudioFrontendChain":
        """Build the standard desktop ASR frontend chain.

        Stage order (mirrors the original AsrAudioFrontendV2 behaviour):
        1. HighpassStage      — DC removal + pre-emphasis
        2. NoiseReductionStage — stationary noise suppression
        3. MusicSuppressionStage — tonal/music bed attenuation
        4. LoudnessStage      — AGC / gain normalisation
        """
        stages: list[object] = [
            HighpassStage(alpha=highpass_alpha, enabled=enabled),
            NoiseReductionStage(
                strength=enhancement_noise_reduce_strength,
                adapt_rate=enhancement_noise_adapt_rate,
                enabled=enabled and enhancement_enabled,
            ),
            MusicSuppressionStage(
                strength=enhancement_music_suppress_strength,
                enabled=enabled and enhancement_enabled,
            ),
            LoudnessStage(
                target_rms=target_rms,
                max_gain=max_gain,
                enabled=enabled,
            ),
        ]
        return cls(stages, enabled=enabled)

    # ------------------------------------------------------------------
    # AEC / reference integration point
    # ------------------------------------------------------------------

    def insert(self, index: int, stage: object) -> None:
        """Insert a stage at *index* in the pipeline.

        Use this to prepend an AEC/reference-aware stage:
            chain.insert(0, MyAECStage(...))
        """
        self._stages.insert(index, stage)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> ChainResult:
        """Run the full stage chain and return a ChainResult."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        input_rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        if not self._enabled or audio.size == 0:
            return ChainResult(
                audio=audio,
                sample_rate=int(sample_rate),
                input_rms=input_rms,
                output_rms=input_rms,
                applied_gain=1.0,
            )

        out = audio
        for stage in self._stages:
            out = stage.process(out, sample_rate)  # type: ignore[attr-defined]

        output_rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0
        applied_gain = (output_rms / input_rms) if input_rms > 1e-9 else 1.0

        return ChainResult(
            audio=out.astype(np.float32, copy=False),
            sample_rate=int(sample_rate),
            input_rms=input_rms,
            output_rms=output_rms,
            applied_gain=applied_gain,
        )

    def reset(self) -> None:
        """Reset all stages (call between utterances / on channel restart)."""
        for stage in self._stages:
            stage.reset()  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self._stages)

    def __repr__(self) -> str:  # pragma: no cover
        names = [type(s).__name__ for s in self._stages]
        return f"AudioFrontendChain(enabled={self._enabled}, stages={names})"


__all__ = ["AudioFrontendChain", "ChainResult"]
