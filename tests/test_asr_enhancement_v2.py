from __future__ import annotations

import unittest

import numpy as np

from app.infra.asr.enhancement_v2 import AsrSpeechEnhancerV2


class AsrEnhancementV2Tests(unittest.TestCase):
    def test_enhancer_detects_tonal_music_bed(self) -> None:
        enhancer = AsrSpeechEnhancerV2(enabled=True, music_suppress_strength=0.35)
        t = np.linspace(0.0, 0.4, 6400, endpoint=False, dtype=np.float32)
        tone = (
            0.05 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.04 * np.sin(2.0 * np.pi * 440.0 * t)
        ).astype(np.float32)

        result = enhancer.process(tone, 16000, speech_ratio=0.18)

        self.assertGreater(result.music_likelihood, 0.2)
        self.assertGreater(result.suppression_ratio, 0.01)

    def test_enhancer_preserves_speechy_burst_energy(self) -> None:
        enhancer = AsrSpeechEnhancerV2(enabled=True)
        signal = np.zeros((6400,), dtype=np.float32)
        t = np.linspace(0.0, 0.08, 1280, endpoint=False, dtype=np.float32)
        burst = (
            0.06 * np.sin(2.0 * np.pi * 180.0 * t)
            + 0.03 * np.sin(2.0 * np.pi * 360.0 * t)
            + 0.01 * np.random.default_rng(7).normal(size=t.shape[0]).astype(np.float32)
        ).astype(np.float32)
        signal[1200:2480] = burst

        result = enhancer.process(signal, 16000, speech_ratio=0.52)

        self.assertGreater(float(np.max(np.abs(result.audio))), 0.015)
        self.assertLess(result.music_likelihood, 0.8)


if __name__ == "__main__":
    unittest.main()
