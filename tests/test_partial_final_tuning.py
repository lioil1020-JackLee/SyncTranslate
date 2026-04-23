from __future__ import annotations

from app.application.audio_router import AudioRouter
from app.application.transcript_service import TranscriptBuffer
from app.domain.runtime_state import StateManager
from app.infra.asr.endpointing_v2 import EndpointSignal
from app.infra.asr.streaming_policy import StreamingDecision
from app.infra.asr.worker_v2 import SourceRuntimeV2
from app.infra.config.schema import AppConfig
from app.infra.asr.backend_v2 import BackendTranscript


class _NoopInputManager:
    def stats(self):
        return {}


class _NoopAsrManager:
    _pipeline_revision = 1

    def configure_pipeline(self, _config, _pipeline_revision: int):
        return None

    def stats(self):
        return {}


class _NoopTranslatorManager:
    pass


class _NoopTtsManager:
    def stats(self):
        return {}

    def is_passthrough_enabled(self, _channel: str) -> bool:
        return False


def _build_router() -> AudioRouter:
    return AudioRouter(
        transcript_buffer=TranscriptBuffer(),
        input_manager=_NoopInputManager(),  # type: ignore[arg-type]
        asr_manager=_NoopAsrManager(),  # type: ignore[arg-type]
        translator_manager=_NoopTranslatorManager(),  # type: ignore[arg-type]
        tts_manager=_NoopTtsManager(),  # type: ignore[arg-type]
        state_manager=StateManager(),
    )


def test_router_accepts_large_append_when_prefix_is_stable() -> None:
    router = _build_router()
    config = AppConfig()
    config.runtime.partial_stability_max_delta_chars = 6
    router.refresh_runtime_config(config)

    previous = "We need to update the deployment pipeline"
    current = "We need to update the deployment pipeline before Friday morning"

    assert router._is_stable_partial_progression(previous, current)


def test_merge_final_with_last_partial_prefers_partial_on_long_shared_prefix_regression() -> None:
    merged = SourceRuntimeV2._merge_final_with_last_partial(
        final_text="We need to update the deployment pipeline",
        last_partial_text="We need to update the deployment pipeline before Friday morning",
    )
    assert merged == "We need to update the deployment pipeline before Friday morning"


def test_merge_final_with_last_partial_drops_repetitive_loop_final() -> None:
    merged = SourceRuntimeV2._merge_final_with_last_partial(
        final_text="所以就是到了親家那邊 所以就是到了親家那邊 所以就是到了親家那邊",
        last_partial_text="所以就是到了親家那邊",
    )
    assert merged == "所以就是到了親家那邊"


def test_short_pause_early_final_is_deferred_for_recent_partial() -> None:
    runtime = SourceRuntimeV2(
        source="local",
        partial_backend=object(),
        final_backend=object(),
        endpointing=object(),  # type: ignore[arg-type]
        partial_interval_ms=480,
        partial_history_seconds=2,
        final_history_seconds=12,
        soft_final_audio_ms=4200,
        pre_roll_ms=320,
        min_partial_audio_ms=320,
        queue_maxsize=32,
        early_final_enabled=True,
        adaptive_enabled=False,
        degradation_enabled=False,
    )
    runtime._last_partial_text = "記者會回答記者的問題"
    runtime._last_partial_emit_ms = 1000

    deferred = runtime._should_defer_early_final(
        now_ms=1240,
        decision=StreamingDecision(emit_final=True, is_early_final=True, reason="pause_turn"),
        signal=EndpointSignal(pause_ms=160.0, speech_active=True, speech_ended=False),
        segment_audio_ms=1400,
    )

    assert deferred is True
    assert runtime._deferred_early_final_until_ms > 1240


def test_merge_final_with_last_partial_keeps_distinct_final_when_overlap_is_low() -> None:
    merged = SourceRuntimeV2._merge_final_with_last_partial(
        final_text="I think this is the better final sentence",
        last_partial_text="Maybe we should try a different draft instead",
    )
    assert merged == "I think this is the better final sentence"


def test_reason_to_prefer_last_partial_reports_only_real_truncation() -> None:
    reason = SourceRuntimeV2._reason_to_prefer_last_partial(
        final_text="We need to update the deployment pipeline",
        last_partial_text="We need to update the deployment pipeline before Friday morning",
    )
    assert reason == "final-substring-regression"

    unrelated_reason = SourceRuntimeV2._reason_to_prefer_last_partial(
        final_text="This is a completely different final result",
        last_partial_text="The partial text wandered somewhere else entirely",
    )
    assert unrelated_reason == ""


def test_empty_final_can_retry_on_shorter_tail() -> None:
    runtime = SourceRuntimeV2(
        source="local",
        partial_backend=object(),
        final_backend=object(),
        endpointing=object(),  # type: ignore[arg-type]
        partial_interval_ms=480,
        partial_history_seconds=2,
        final_history_seconds=12,
        soft_final_audio_ms=4200,
        pre_roll_ms=320,
        min_partial_audio_ms=320,
        queue_maxsize=32,
        early_final_enabled=True,
        adaptive_enabled=False,
        degradation_enabled=False,
    )
    runtime._segment_sample_rate = 16000
    runtime._segment_chunks = [
        __import__("numpy").ones((16000 * 14,), dtype=__import__("numpy").float32),
    ]
    runtime._last_partial_text = "there is still a valid partial"

    calls: list[int] = []

    def _fake_transcribe_backend(_backend, *, method_name: str, audio, sample_rate: int):
        calls.append(int(audio.shape[0]))
        return BackendTranscript(text="recovered final", is_final=True, detected_language="en")

    runtime._transcribe_backend = _fake_transcribe_backend  # type: ignore[method-assign]
    first = BackendTranscript(text="", is_final=True, detected_language="en")

    retried = runtime._retry_empty_final_with_shorter_tail(
        result=first,
        audio=runtime._limited_audio(runtime._final_history_seconds),
        sample_rate=16000,
    )

    assert retried.text == "recovered final"
    assert len(calls) == 1
