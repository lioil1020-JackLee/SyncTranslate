from __future__ import annotations

import unittest

import numpy as np

from app.infra.asr.frontend_v2 import AsrAudioFrontendV2


class AsrFrontendV2Tests(unittest.TestCase):
    def test_frontend_prefers_dominant_stereo_channel(self) -> None:
        frontend = AsrAudioFrontendV2()
        t = np.linspace(0.0, 0.02, 960, endpoint=False, dtype=np.float32)
        stereo = np.zeros((960, 2), dtype=np.float32)
        stereo[:, 1] = 0.4 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

        chunk = frontend.process(stereo, 48000)

        self.assertEqual(chunk.audio.shape, (960,))
        self.assertGreater(float(np.max(np.abs(chunk.audio))), 0.03)

    def test_frontend_applies_mild_gain_for_quiet_speech(self) -> None:
        frontend = AsrAudioFrontendV2()
        t = np.linspace(0.0, 0.1, 1600, endpoint=False, dtype=np.float32)
        quiet = 0.005 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

        chunk = frontend.process(quiet, 16000)

        self.assertGreater(chunk.output_rms, chunk.input_rms)
        self.assertGreater(chunk.applied_gain, 1.0)

    def test_frontend_reports_lower_speech_ratio_for_sparse_signal(self) -> None:
        frontend = AsrAudioFrontendV2()
        signal = np.zeros((1600,), dtype=np.float32)
        signal[200:320] = 0.03

        chunk = frontend.process(signal, 16000)

        self.assertLess(chunk.speech_ratio, 0.5)

    def test_frontend_reports_high_speech_ratio_for_continuous_voiced_signal(self) -> None:
        frontend = AsrAudioFrontendV2(enhancement_enabled=False)
        t = np.linspace(0.0, 0.4, 6400, endpoint=False, dtype=np.float32)
        envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.0 * t).astype(np.float32)
        signal = (0.04 * envelope * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)

        chunk = frontend.process(signal, 16000)

        self.assertGreater(chunk.speech_ratio, 0.7)

    def test_frontend_surfaces_enhancement_stats(self) -> None:
        frontend = AsrAudioFrontendV2()
        t = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
        music_bed = 0.04 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

        chunk = frontend.process(music_bed, 16000)

        self.assertGreaterEqual(chunk.noise_floor_rms, 0.0)
        self.assertGreaterEqual(chunk.music_likelihood, 0.0)


if __name__ == "__main__":
    unittest.main()
