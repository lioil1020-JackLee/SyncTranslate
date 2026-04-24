from __future__ import annotations

import unittest

import numpy as np

from app.infra.asr.resampling import resample_audio


class AsrResamplingTests(unittest.TestCase):
    def test_integer_downsample_preserves_speech_band_tone(self) -> None:
        sample_rate = 48000
        t = np.linspace(0.0, 0.2, int(sample_rate * 0.2), endpoint=False, dtype=np.float32)
        tone = 0.4 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)

        resampled = resample_audio(tone, sample_rate=sample_rate, target_rate=16000)

        self.assertEqual(resampled.shape[0], 3200)
        self.assertGreater(float(np.sqrt(np.mean(np.square(resampled)))), 0.20)

    def test_integer_downsample_reduces_alias_prone_high_frequency_tone(self) -> None:
        sample_rate = 48000
        t = np.linspace(0.0, 0.2, int(sample_rate * 0.2), endpoint=False, dtype=np.float32)
        tone = 0.4 * np.sin(2.0 * np.pi * 12000.0 * t).astype(np.float32)

        resampled = resample_audio(tone, sample_rate=sample_rate, target_rate=16000)

        self.assertLess(float(np.sqrt(np.mean(np.square(resampled)))), 0.08)


if __name__ == "__main__":
    unittest.main()
