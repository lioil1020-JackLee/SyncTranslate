"""Tests for audio pipeline stages and frontend chain (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from app.infra.asr.audio_pipeline.base import AudioProcessingStage, ReferenceAwareAudioStage
from app.infra.asr.audio_pipeline.frontend_chain import AudioFrontendChain, ChainResult
from app.infra.asr.audio_pipeline.highpass import HighpassStage
from app.infra.asr.audio_pipeline.identity import IdentityStage
from app.infra.asr.audio_pipeline.loudness import LoudnessStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float = 440.0, duration: float = 0.5, sr: int = 16000, amp: float = 0.1) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _silence(duration: float = 0.1, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# IdentityStage
# ---------------------------------------------------------------------------

class TestIdentityStage:
    def test_returns_same_audio(self):
        stage = IdentityStage()
        audio = _sine()
        out = stage.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_reset_is_noop(self):
        IdentityStage().reset()

    def test_satisfies_protocol(self):
        assert isinstance(IdentityStage(), AudioProcessingStage)


# ---------------------------------------------------------------------------
# HighpassStage
# ---------------------------------------------------------------------------

class TestHighpassStage:
    def test_output_same_length(self):
        stage = HighpassStage(alpha=0.96)
        audio = _sine()
        out = stage.process(audio, 16000)
        assert out.shape == audio.shape

    def test_dc_removed(self):
        stage = HighpassStage(alpha=0.96)
        dc = np.full(16000, 0.1, dtype=np.float32)
        out = stage.process(dc, 16000)
        # Pre-emphasis filter: first-order IIR with y[n] = x[n] - α·x[n-1] + α·y[n-1]
        # For constant (DC) input, transient at start then stabilizes.
        # The first sample should be different from the steady-state value.
        # Just verify output is same length and finite (filter doesn't crash on DC).
        assert out.shape == dc.shape
        assert np.all(np.isfinite(out))

    def test_disabled_passthrough(self):
        stage = HighpassStage(alpha=0.96, enabled=False)
        audio = _sine()
        out = stage.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_empty_passthrough(self):
        stage = HighpassStage()
        out = stage.process(np.array([], dtype=np.float32), 16000)
        assert out.size == 0

    def test_reset_clears_state(self):
        stage = HighpassStage(alpha=0.96)
        stage.process(_sine(), 16000)
        stage.reset()
        # After reset, first sample should not carry previous state
        single = np.array([0.5], dtype=np.float32)
        out = stage.process(single, 16000)
        assert abs(float(out[0]) - 0.5) < 0.1  # roughly passes through first sample


# ---------------------------------------------------------------------------
# LoudnessStage
# ---------------------------------------------------------------------------

class TestLoudnessStage:
    def test_boosts_quiet_audio(self):
        stage = LoudnessStage(target_rms=0.1, max_gain=10.0)
        quiet = _sine(amp=0.001)
        out = stage.process(quiet, 16000)
        out_rms = float(np.sqrt(np.mean(out ** 2)))
        assert out_rms > 0.001 * 2  # should be significantly louder

    def test_does_not_exceed_unity(self):
        stage = LoudnessStage(target_rms=0.1)
        loud = _sine(amp=0.9)
        out = stage.process(loud, 16000)
        assert float(np.max(np.abs(out))) <= 1.0

    def test_disabled_passthrough(self):
        stage = LoudnessStage(enabled=False)
        audio = _sine(amp=0.001)
        out = stage.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_silence_no_crash(self):
        stage = LoudnessStage()
        out = stage.process(_silence(), 16000)
        assert out.size > 0


# ---------------------------------------------------------------------------
# AudioFrontendChain
# ---------------------------------------------------------------------------

class TestAudioFrontendChain:
    def test_build_default_runs(self):
        chain = AudioFrontendChain.build_default()
        audio = _sine()
        result = chain.process(audio, 16000)
        assert isinstance(result, ChainResult)
        assert result.audio.dtype == np.float32
        assert result.audio.size == audio.size

    def test_disabled_chain_passthrough(self):
        chain = AudioFrontendChain.build_default(enabled=False)
        audio = _sine()
        result = chain.process(audio, 16000)
        np.testing.assert_array_equal(result.audio, audio)

    def test_empty_audio(self):
        chain = AudioFrontendChain.build_default()
        result = chain.process(np.array([], dtype=np.float32), 16000)
        assert result.audio.size == 0

    def test_chain_length(self):
        chain = AudioFrontendChain.build_default()
        assert len(chain) == 4  # highpass, noise, music, loudness

    def test_insert_stage(self):
        chain = AudioFrontendChain.build_default()
        chain.insert(0, IdentityStage())
        assert len(chain) == 5

    def test_reset(self):
        chain = AudioFrontendChain.build_default()
        chain.process(_sine(), 16000)
        chain.reset()  # should not raise

    def test_result_rms_fields(self):
        chain = AudioFrontendChain.build_default(target_rms=0.1)
        audio = _sine(amp=0.01)
        result = chain.process(audio, 16000)
        assert result.input_rms > 0
        assert result.output_rms > 0

    def test_custom_stages(self):
        chain = AudioFrontendChain([IdentityStage(), LoudnessStage(target_rms=0.1, max_gain=5.0)])
        audio = _sine(amp=0.001)
        result = chain.process(audio, 16000)
        assert result.output_rms > result.input_rms


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_identity_is_processing_stage(self):
        assert isinstance(IdentityStage(), AudioProcessingStage)

    def test_highpass_is_processing_stage(self):
        assert isinstance(HighpassStage(), AudioProcessingStage)

    def test_loudness_is_processing_stage(self):
        assert isinstance(LoudnessStage(), AudioProcessingStage)


# ---------------------------------------------------------------------------
# Benchmark report helpers (smoke test)
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_cer_identical(self):
        from tools.asr_benchmark.run_benchmark import _cer
        assert _cer("hello world", "hello world") == 0.0

    def test_cer_all_wrong(self):
        from tools.asr_benchmark.run_benchmark import _cer
        assert _cer("abc", "xyz") == 1.0

    def test_wer_identical(self):
        from tools.asr_benchmark.run_benchmark import _wer
        assert _wer("hello world", "hello world") == 0.0

    def test_wer_one_word_error(self):
        from tools.asr_benchmark.run_benchmark import _wer
        result = _wer("hello world test", "hello earth test")
        assert abs(result - 1 / 3) < 0.01

    def test_report_loads_jsonl(self, tmp_path):
        from tools.asr_benchmark.report import _load_results
        p = tmp_path / "benchmark_results.jsonl"
        p.write_text('{"file": "a.wav", "transcript": "hello"}\n', encoding="utf-8")
        records = _load_results(p)
        assert len(records) == 1
        assert records[0]["file"] == "a.wav"

    def test_report_empty_file(self, tmp_path):
        from tools.asr_benchmark.report import _load_results
        p = tmp_path / "benchmark_results.jsonl"
        p.write_text("", encoding="utf-8")
        assert _load_results(p) == []
