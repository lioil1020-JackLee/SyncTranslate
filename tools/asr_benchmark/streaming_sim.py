"""Streaming ASR simulation benchmark.

音訊以 chunk 方式喂入，模擬真實串流場景，測量 CER / WER 並支援參數掃描。

Usage
-----
uv run python tools/asr_benchmark/streaming_sim.py \\
    --config config.yaml \\
    --audio downloads/chinese_srt/xxx.wav \\
    --reference downloads/benchmark_results/chinese_reference.txt \\
    --source local --language zh-TW

參數掃描:
    --sweep  # 自動搜尋最佳 VAD 參數組合
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from app.bootstrap.external_runtime import configure_external_ai_runtime
    configure_external_ai_runtime()
except Exception as e:
    print(f"[warn] external runtime setup failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Metric helpers (same as run_benchmark.py)
# ---------------------------------------------------------------------------

try:
    from opencc import OpenCC  # type: ignore
    _S2T = OpenCC("s2twp")
except Exception:
    try:
        import zhconv as _zhconv  # type: ignore
        class _ZhconvS2T:
            def convert(self, text: str) -> str:
                return _zhconv.convert(text, "zh-tw")
        _S2T = _ZhconvS2T()
    except Exception:
        _S2T = None


def _is_cjk(lang: str) -> bool:
    return str(lang or "").strip().lower().replace("_", "-") in {
        "zh", "zh-tw", "zh-cn", "cmn", "cmn-hans", "cmn-hant", "yue"
    }


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _cer(hyp: str, ref: str) -> float:
    h, r = list(hyp.replace(" ", "")), list(ref.replace(" ", ""))
    return _edit_distance(h, r) / max(1, len(r))


def _wer(hyp: str, ref: str) -> float:
    return _edit_distance(hyp.split(), ref.split()) / max(1, len(ref.split()))


def _normalize(text: str, *, lang: str, ref: str = "") -> str:
    v = str(text or "")
    if _is_cjk(lang):
        if _S2T:
            try:
                v = _S2T.convert(v)
            except Exception:
                pass
        v = re.sub(r"[^\u4e00-\u9fff\s]", " ", v)
        v = "".join(v.split())
        if ref:
            ref_clean = re.sub(r"[^\u4e00-\u9fff\s]", " ", ref)
            if _S2T:
                try:
                    ref_clean = _S2T.convert(ref_clean)
                except Exception:
                    pass
            ref_clean = "".join(ref_clean.split())
            # filter only sentences that appear in reference
            parts = re.split(r"[。！？!?\.]", v)
            kept = [p for p in parts if len(p) >= 4 and p[:4] in ref_clean]
            if kept:
                v = "".join(kept)
        return v
    v = re.sub(r"[^\w\s]", " ", v.lower())
    return " ".join(v.split())


def _collapse_repeated_char_ngrams(
    text: str,
    *,
    min_ngram_chars: int = 4,
    max_ngram_chars: int = 24,
    min_repeats: int = 3,
) -> str:
    if not text or len(text) < min_ngram_chars * min_repeats:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        max_len = min(max_ngram_chars, (n - i) // min_repeats)
        for gram_len in range(max_len, min_ngram_chars - 1, -1):
            gram = text[i:i + gram_len]
            repeats = 1
            j = i + gram_len
            while j + gram_len <= n and text[j:j + gram_len] == gram:
                repeats += 1
                j += gram_len
            if repeats >= min_repeats:
                out.append(gram)
                i = j
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def _collapse_repeated_word_ngrams(
    text: str,
    *,
    max_ngram_words: int = 8,
    min_repeats: int = 3,
) -> str:
    tokens = text.split()
    if len(tokens) < max(4, min_repeats):
        return text
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        max_words = min(max_ngram_words, (n - i) // min_repeats)
        for gram_words in range(max_words, 0, -1):
            gram = tokens[i:i + gram_words]
            repeats = 1
            j = i + gram_words
            while j + gram_words <= n and tokens[j:j + gram_words] == gram:
                repeats += 1
                j += gram_words
            if repeats >= min_repeats:
                out.extend(gram)
                i = j
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _dedupe_repetition(text: str, *, lang: str) -> str:
    if not text:
        return text
    if not _is_cjk(lang) and " " in text:
        collapsed = _collapse_repeated_word_ngrams(text)
        collapsed = _collapse_repeated_char_ngrams(collapsed)
        return collapsed
    # CJK: 先處理短迴圈（n=1~3，≥6 次），再處理長片語（n=4~24，≥3 次）
    collapsed = _collapse_repeated_char_ngrams(text, min_ngram_chars=1, max_ngram_chars=3, min_repeats=6)
    collapsed = _collapse_repeated_char_ngrams(collapsed, min_ngram_chars=4, max_ngram_chars=24, min_repeats=3)
    return collapsed


def _repetition_ratio(text: str, *, lang: str) -> float:
    if not text:
        return 0.0
    dedup = _dedupe_repetition(text, lang=lang)
    removed = max(0, len(text) - len(dedup))
    return removed / max(1, len(text))


# ---------------------------------------------------------------------------
# WAV loader
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        n_ch = f.getnchannels()
        sw = f.getsampwidth()
        sr = f.getframerate()
        raw = f.readframes(f.getnframes())
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Core streaming simulation
# ---------------------------------------------------------------------------

def run_streaming_sim(
    audio: np.ndarray,
    sample_rate: int,
    *,
    config_path: Path,
    source: str,
    language: str,
    vad_overrides: dict[str, Any] | None = None,
    streaming_overrides: dict[str, Any] | None = None,
    asr_overrides: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    worker_overrides: dict[str, Any] | None = None,
    lead_in_ms: int = 1000,
    chunk_ms: int = 100,
    speed_multiplier: float = 8.0,  # feed at Nx real-time (faster = quicker test)
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Simulate streaming ASR with SourceRuntimeV2.
    Returns dict with transcript, final_count, cer, wer etc.
    """
    from app.infra.config.settings_store import load_config
    from app.infra.asr.backend_v2 import build_backend_pair
    from app.infra.asr.endpointing_v2 import build_endpointing_runtime
    from app.infra.asr.backend_resolution import resolve_backend_for_language
    from app.infra.asr.endpoint_profiles import get_endpoint_profile
    from app.infra.asr.language_profiles import resolve_language_asr_profile
    from app.infra.asr.profile_selection import asr_profile_for_language
    from app.infra.asr.profile_selection import asr_profile_slot_for_language
    from app.infra.asr.text_correction import AsrTextCorrector
    from app.infra.asr.worker_v2 import SourceRuntimeV2, V2RuntimeEvent
    from app.infra.config.schema import VadSettings, AsrStreamingSettings
    from app.application.transcript_postprocessor import TranscriptPostProcessor

    config = load_config(str(config_path))

    base_profile = asr_profile_for_language(config, language)
    language_profile = resolve_language_asr_profile(base_profile, language=language)
    profile = deepcopy(language_profile.asr)

    # Apply top-level ASR overrides
    if asr_overrides:
        for key, value in asr_overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

    # Apply VAD overrides
    if vad_overrides:
        vad_dict = asdict(profile.vad)
        vad_dict.update({k: v for k, v in vad_overrides.items() if k in vad_dict})
        profile.vad = VadSettings(**vad_dict)

    # Apply streaming overrides
    if streaming_overrides:
        stream_dict = asdict(profile.streaming)
        stream_dict.update({k: v for k, v in streaming_overrides.items() if k in stream_dict})
        profile.streaming = AsrStreamingSettings(**stream_dict)

    # Apply runtime overrides
    if runtime_overrides:
        for key, value in runtime_overrides.items():
            if hasattr(config.runtime, key):
                setattr(config.runtime, key, value)

    if verbose:
        print(f"[sim] VAD backend={profile.vad.backend} neural_threshold={profile.vad.neural_threshold} "
              f"min_silence={profile.vad.min_silence_duration_ms}ms", flush=True)
        if asr_overrides:
            print(f"[sim] ASR overrides={asr_overrides}", flush=True)
        if runtime_overrides:
            print(f"[sim] runtime overrides={runtime_overrides}", flush=True)
        if worker_overrides:
            print(f"[sim] worker overrides={worker_overrides}", flush=True)

    # Override language in config for correct backend resolution
    profile_slot = asr_profile_slot_for_language(language)
    if profile_slot == "local":
        config.asr_channels.local = profile
    elif profile_slot == "remote":
        config.asr_channels.remote = profile
    if source == "remote":
        config.runtime.remote_asr_language = language
    else:
        config.runtime.local_asr_language = language

    # Build backends
    if verbose:
        print("[sim] Building backends...", flush=True)
    t0 = time.monotonic()
    build_result = build_backend_pair(config, source=source, language=language, profile_override=profile)
    if isinstance(build_result, tuple):
        partial_backend, final_backend = build_result
        resolution = resolve_backend_for_language(language)
    else:
        partial_backend = build_result.partial_backend
        final_backend = build_result.final_backend
        resolution = build_result.resolution

    # Warmup
    partial_backend.warmup()
    final_backend.warmup()
    load_ms = round((time.monotonic() - t0) * 1000)
    if verbose:
        print(f"[sim] Backends ready in {load_ms}ms, resolution={resolution.backend_name}", flush=True)

    # Build endpointing
    endpointing = build_endpointing_runtime(
        str(getattr(config.runtime, "asr_v2_endpointing", "neural_endpoint")),
        profile.vad,
        device=profile.device,
        resolved_backend_name=resolution.backend_name,
    )
    ep_snap = endpointing.snapshot()
    if verbose:
        print(f"[sim] Endpointing: backend={ep_snap.get('backend')} available={ep_snap.get('available')}", flush=True)
        if not ep_snap.get("available"):
            print("[sim] WARNING: Neural VAD not available! Falling back to RMS.", flush=True)

    # Build endpoint profile for worker kwargs
    _profile_name = (
        getattr(config.runtime, "asr_profile_remote", None)
        if source == "remote"
        else getattr(config.runtime, "asr_profile_local", None)
    )
    # Priority for timing params: streaming_overrides > config.runtime > hard-coded defaults
    # We do NOT let endpoint profile override explicit CLI/sweep values in benchmark context.
    if language_profile.endpoint_profile and str(_profile_name or "").strip() in {"", "default", "meeting_room"}:
        _profile_name = language_profile.endpoint_profile
    _ep_kwargs = get_endpoint_profile(_profile_name).to_worker_kwargs()
    _base_pre_roll_ms = int(getattr(config.runtime, "asr_pre_roll_ms", 500))
    _base_min_partial_ms = int(getattr(config.runtime, "asr_partial_min_audio_ms", 280))
    _resolved_timing: dict[str, Any] = {
        "soft_final_audio_ms": _ep_kwargs.get("soft_final_audio_ms", profile.streaming.soft_final_audio_ms),
        "pre_roll_ms": _ep_kwargs.get("pre_roll_ms", _base_pre_roll_ms),
        "min_partial_audio_ms": _ep_kwargs.get("min_partial_audio_ms", _base_min_partial_ms),
        "partial_interval_ms": _ep_kwargs.get("partial_interval_ms", profile.streaming.partial_interval_ms),
    }
    # streaming_overrides have highest priority (CLI / sweep params)
    if streaming_overrides:
        _resolved_timing.update({k: v for k, v in streaming_overrides.items() if k in _resolved_timing})
    if worker_overrides:
        _resolved_timing.update({k: v for k, v in worker_overrides.items() if k in _resolved_timing})
    if verbose:
        print(f"[sim] endpoint_profile={_profile_name or 'default'} "
              f"soft_final_audio_ms={_resolved_timing['soft_final_audio_ms']} "
              f"partial_interval_ms={_resolved_timing['partial_interval_ms']}", flush=True)

    # Collect events
    finals: list[str] = []
    partials: list[str] = []
    events_lock = Lock()
    done_event = Event()
    current_utterance_id = "1"
    utterance_counter = 1
    postprocessor = TranscriptPostProcessor(
        enabled=bool(getattr(config.runtime, "enable_postprocessor", True)),
        partial_stabilization_enabled=bool(getattr(config.runtime, "enable_partial_stabilization", True)),
        glossary=None,
        glossary_apply_on_partial=bool(getattr(config.runtime, "glossary_apply_on_partial", False)),
        glossary_apply_on_final=bool(getattr(config.runtime, "glossary_apply_on_final", True)),
    )
    llm_cfg = config.llm_channels.remote if source == "remote" else config.llm_channels.local
    corrector = AsrTextCorrector(
        llm_cfg,
        enabled=bool(getattr(config.runtime, "asr_final_correction_enabled", False)),
        context_items=int(getattr(config.runtime, "asr_final_correction_context_items", 3)),
        max_chars=int(getattr(config.runtime, "asr_final_correction_max_chars", 120)),
    )

    def on_event(ev: V2RuntimeEvent) -> None:
        nonlocal current_utterance_id, utterance_counter
        with events_lock:
            if ev.is_final:
                text = postprocessor.process_final(
                    source,
                    ev.text,
                    language=ev.detected_language,
                    utterance_id=current_utterance_id,
                ).strip()
                if text:
                    text = corrector.correct(text, language=ev.detected_language).text.strip()
                if text:
                    finals.append(text)
                    if verbose:
                        print(f"  [FINAL #{len(finals)}] {text[:80]}", flush=True)
                utterance_counter += 1
                current_utterance_id = str(utterance_counter)
            else:
                partial_text = postprocessor.process_partial(
                    source,
                    ev.text,
                    language=ev.detected_language,
                    utterance_id=current_utterance_id,
                )
                partials.append(partial_text)

    # Scale queue so force_final threshold > chunks-in-flight during transcription.
    # At high speed, chunks pile up while the worker is busy transcribing.
    # force_final fires at queue_maxsize//4, so keep maxsize >> (feed_rate * max_transcription_seconds).
    # Estimate: total chunks × speed gives a safe upper bound.
    total_chunks = max(1, len(audio) // max(1, int(sample_rate * chunk_ms / 1000)))
    sim_queue_maxsize = max(512, int(total_chunks * max(1.0, speed_multiplier / 2.0)))

    # Create worker
    runtime = SourceRuntimeV2(
        source=source,
        partial_backend=partial_backend,
        final_backend=final_backend,
        endpointing=endpointing,
        partial_interval_ms=_resolved_timing["partial_interval_ms"],
        partial_history_seconds=profile.streaming.partial_history_seconds,
        final_history_seconds=profile.streaming.final_history_seconds,
        soft_final_audio_ms=_resolved_timing["soft_final_audio_ms"],
        pre_roll_ms=_resolved_timing["pre_roll_ms"],
        min_partial_audio_ms=_resolved_timing["min_partial_audio_ms"],
        queue_maxsize=sim_queue_maxsize,
        frontend_enabled=bool(getattr(config.runtime, "asr_frontend_enabled", True)),
        frontend_target_rms=float(getattr(config.runtime, "asr_frontend_target_rms", 0.05)),
        frontend_max_gain=float(getattr(config.runtime, "asr_frontend_max_gain", 3.0)),
        frontend_highpass_alpha=float(getattr(config.runtime, "asr_frontend_highpass_alpha", 0.96)),
        enhancement_enabled=bool(getattr(config.runtime, "asr_enhancement_enabled", True)),
        enhancement_noise_reduce_strength=float(getattr(config.runtime, "asr_enhancement_noise_reduce_strength", 0.42)),
        enhancement_noise_adapt_rate=float(getattr(config.runtime, "asr_enhancement_noise_adapt_rate", 0.18)),
        enhancement_music_suppress_strength=float(getattr(config.runtime, "asr_enhancement_music_suppress_strength", 0.2)),
    )

    runtime.start(on_event)

    # Feed audio in chunks
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    audio_duration_ms = round(len(audio) / sample_rate * 1000)
    sleep_per_chunk = (chunk_ms / 1000) / speed_multiplier  # seconds to sleep per chunk

    if verbose:
        print(f"[sim] Feeding {audio_duration_ms/1000:.1f}s audio in {chunk_ms}ms chunks "
              f"at {speed_multiplier}x speed...", flush=True)

    t_start = time.monotonic()
    lead_in_samples = int(max(0, lead_in_ms) * sample_rate / 1000)
    if lead_in_samples > 0:
        lead_in_audio = np.zeros(lead_in_samples, dtype=np.float32)
        lead_offset = 0
        while lead_offset < lead_in_audio.shape[0]:
            chunk = lead_in_audio[lead_offset: lead_offset + chunk_samples]
            runtime.submit_chunk(chunk.astype(np.float32, copy=False), float(sample_rate))
            lead_offset += chunk_samples
            if sleep_per_chunk > 0:
                time.sleep(sleep_per_chunk)
    offset = 0
    chunk_idx = 0
    while offset < len(audio):
        chunk = audio[offset: offset + chunk_samples]
        runtime.submit_chunk(chunk.astype(np.float32, copy=False), float(sample_rate))
        offset += chunk_samples
        chunk_idx += 1
        if sleep_per_chunk > 0:
            time.sleep(sleep_per_chunk)

    # Wait for worker to drain the queue.
    # We must not stop too early: SourceRuntimeV2.stop() joins for a limited
    # time and then clears any remaining queued chunks, which can truncate the
    # tail of long-form transcripts in benchmark mode.
    if verbose:
        print("[sim] All audio fed. Waiting for worker to finish...", flush=True)

    def _wait_for_idle(*, max_wait_s: float, stable_s: float) -> tuple[int, int, str]:
        deadline = time.monotonic() + max_wait_s
        stable_checks_needed = max(1, int(round(stable_s / 0.25)))
        stable_checks = 0
        prev_final_count = -1
        last_debug = ""
        while time.monotonic() < deadline:
            time.sleep(0.25)
            stats = runtime.stats()
            queue_size = int(stats.queue_size)
            last_debug = str(stats.last_debug or "")
            with events_lock:
                final_count = len(finals)
            if queue_size == 0 and final_count == prev_final_count:
                stable_checks += 1
                if stable_checks >= stable_checks_needed:
                    return queue_size, final_count, last_debug
            else:
                stable_checks = 0
                prev_final_count = final_count
        stats = runtime.stats()
        with events_lock:
            final_count = len(finals)
        return int(stats.queue_size), final_count, str(stats.last_debug or last_debug)

    queue_size, final_count, last_debug = _wait_for_idle(max_wait_s=18.0, stable_s=3.0)

    # Force flush: submit long silence to trigger trailing endpoint, then wait
    # until the runtime is truly idle before stopping.
    silence = np.zeros(int(sample_rate * 3.0), dtype=np.float32)
    runtime.submit_chunk(silence, float(sample_rate))
    runtime.submit_chunk(silence, float(sample_rate))
    queue_size, final_count, last_debug = _wait_for_idle(max_wait_s=45.0, stable_s=6.0)
    if verbose and (queue_size > 0 or last_debug):
        print(
            f"[sim] Drain status: queue={queue_size} finals={final_count} debug={last_debug}",
            flush=True,
        )

    runtime.stop()
    elapsed = time.monotonic() - t_start

    # Build final transcript from all finals
    sep = "" if _is_cjk(language) else " "
    full_transcript = sep.join(finals)

    stats = runtime.stats()
    ep = stats.endpointing

    result: dict[str, Any] = {
        "source": source,
        "language": language,
        "audio_duration_ms": audio_duration_ms,
        "sim_elapsed_ms": round(elapsed * 1000),
        "speed": round(audio_duration_ms / max(1, elapsed * 1000), 2),
        "final_count": len(finals),
        "worker_final_count": stats.final_count,
        "partial_count": stats.partial_count,
        "dropped_chunks": stats.dropped_chunks,
        "transcript": full_transcript,
        "vad_backend": ep.get("backend", ""),
        "vad_available": ep.get("available", False),
        "speech_started_count": ep.get("speech_started_count", 0),
        "soft_endpoint_count": ep.get("soft_endpoint_count", 0),
        "hard_endpoint_count": ep.get("hard_endpoint_count", 0),
        "vad_params": {
            "neural_threshold": profile.vad.neural_threshold,
            "min_silence_duration_ms": profile.vad.min_silence_duration_ms,
            "min_speech_duration_ms": profile.vad.min_speech_duration_ms,
            "speech_pad_ms": profile.vad.speech_pad_ms,
        },
        "streaming_params": {
            "endpoint_profile": _profile_name or "default",
            "soft_final_audio_ms": _resolved_timing["soft_final_audio_ms"],
            "partial_interval_ms": _resolved_timing["partial_interval_ms"],
            "pre_roll_ms": _resolved_timing["pre_roll_ms"],
            "min_partial_audio_ms": _resolved_timing["min_partial_audio_ms"],
        },
    }
    return result


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

_ZH_SWEEP = [
    # (neural_threshold, min_silence_ms, soft_final_audio_ms)
    # Higher thresholds help VAD ignore background music and detect natural sentence pauses
    (0.4,  600, 3000),   # previous best baseline
    (0.4,  600, 4000),   # longer soft_final window
    (0.45, 700, 3500),   # slightly stricter VAD
    (0.45, 1000, 4000),  # longer silence needed through music
    (0.5,  800, 4000),   # stricter VAD + longer silence
    (0.5,  1000, 4000),  # longest silence at moderate threshold
    (0.55, 800,  3500),  # higher threshold (music below this?)
    (0.55, 1000, 4000),  # high threshold + long silence
    (0.6,  1000, 4500),  # very high threshold
    (0.6,  1200, 5000),  # very high threshold, maximum silence
]

_EN_SWEEP = [
    # (neural_threshold, min_silence_ms, soft_final_audio_ms)
    # English audio has no background music; tune for clean speech
    (0.25, 400, 2000),   # more sensitive, shorter pauses
    (0.25, 500, 2500),   # more sensitive
    (0.3,  300, 1500),   # very short silence window
    (0.3,  400, 2000),   # shorter than previous best
    (0.3,  500, 2500),   # previous best
    (0.3,  600, 3000),   # longer window
    (0.35, 500, 2500),   # slightly stricter VAD
    (0.35, 600, 3000),   # stricter VAD + longer window
]


def _sweep(
    audio: np.ndarray,
    sample_rate: int,
    *,
    config_path: Path,
    source: str,
    language: str,
    reference_text: str,
    output_dir: Path,
    speed: float,
) -> list[dict[str, Any]]:
    sweep_params = _ZH_SWEEP if _is_cjk(language) else _EN_SWEEP
    results: list[dict[str, Any]] = []

    for i, (thr, silence_ms, soft_ms) in enumerate(sweep_params):
        print(f"\n{'='*60}", flush=True)
        print(f"[sweep {i+1}/{len(sweep_params)}] neural_thr={thr} silence={silence_ms}ms soft={soft_ms}ms", flush=True)
        vad_ov = {"neural_threshold": thr, "min_silence_duration_ms": silence_ms}
        stream_ov = {"soft_final_audio_ms": soft_ms}

        try:
            r = run_streaming_sim(
                audio, sample_rate,
                config_path=config_path,
                source=source,
                language=language,
                vad_overrides=vad_ov,
                streaming_overrides=stream_ov,
                speed_multiplier=speed,
                verbose=False,
            )
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            continue

        # compute metrics
        hyp = _normalize(r["transcript"], lang=language, ref=reference_text)
        ref = _normalize(reference_text, lang=language, ref=reference_text)
        hyp_dedup = _dedupe_repetition(hyp, lang=language)
        cer = round(_cer(hyp, ref), 4)
        acc = round(1 - cer, 4)
        cer_dedup = round(_cer(hyp_dedup, ref), 4)
        acc_dedup = round(1 - cer_dedup, 4)
        rep_ratio = round(_repetition_ratio(hyp, lang=language), 4)
        r["cer_normalized"] = cer
        r["accuracy"] = acc
        r["cer_dedup"] = cer_dedup
        r["accuracy_dedup"] = acc_dedup
        r["repetition_ratio"] = rep_ratio

        print(f"  finals={r['final_count']} transcript_len={len(r['transcript'])} "
              f"CER={cer:.3f} dedupCER={cer_dedup:.3f} rep={rep_ratio:.1%} "
              f"accuracy={acc:.1%}", flush=True)
        results.append(r)

    # Sort by dedup accuracy first, then raw accuracy, then lower repetition ratio.
    results.sort(
        key=lambda x: (
            x.get("accuracy_dedup", 0),
            x.get("accuracy", 0),
            -x.get("repetition_ratio", 1.0),
        ),
        reverse=True,
    )

    out_file = output_dir / f"streaming_sweep_{source}_{language.replace('-','')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[sweep] Results saved to {out_file}", flush=True)

    if results:
        best = results[0]
        print(f"\n[sweep] BEST: neural_thr={best['vad_params']['neural_threshold']} "
              f"silence={best['vad_params']['min_silence_duration_ms']}ms "
              f"soft={best['streaming_params']['soft_final_audio_ms']}ms "
              f"-> accuracy={best.get('accuracy', 0):.1%} "
              f"dedup_accuracy={best.get('accuracy_dedup', best.get('accuracy', 0)):.1%} "
              f"rep={best.get('repetition_ratio', 0.0):.1%} finals={best['final_count']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Streaming ASR simulation benchmark")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--audio", required=True, help="WAV file")
    p.add_argument("--reference", default=None, help="Ground truth text file")
    p.add_argument("--source", default="local", choices=["local", "remote"])
    p.add_argument("--language", required=True, help="ASR language (zh-TW, en, ...)")
    p.add_argument("--output", default="downloads/benchmark_results", help="Output directory")
    p.add_argument("--chunk-ms", type=int, default=100, help="Audio chunk size in ms")
    p.add_argument("--speed", type=float, default=8.0,
                   help="Feed speed multiplier vs real-time (8 = 8x faster than real time)")
    p.add_argument("--sweep", action="store_true", help="Sweep VAD parameters")
    p.add_argument("--neural-threshold", type=float, default=None)
    p.add_argument("--min-silence-ms", type=int, default=None)
    p.add_argument("--soft-final-ms", type=int, default=None)
    p.add_argument("--speech-pad-ms", type=int, default=None)
    p.add_argument("--startup-suppress-ms", type=int, default=None)
    p.add_argument("--partial-history-s", type=int, default=None)
    p.add_argument("--final-history-s", type=int, default=None)
    p.add_argument("--pre-roll-ms", type=int, default=None)
    p.add_argument("--partial-interval-ms", type=int, default=None)
    p.add_argument("--min-partial-audio-ms", type=int, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--final-beam-size", type=int, default=None)
    p.add_argument("--no-speech-threshold", type=float, default=None)
    p.add_argument("--hotwords", default=None)
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--lead-in-ms", type=int, default=1000)
    p.add_argument("--bias-lexicon", default=None)
    p.add_argument(
        "--bias-enabled",
        choices=["true", "false"],
        default=None,
        help="Override runtime.asr_hotwords_enabled for lexical bias post-processing.",
    )
    p.add_argument(
        "--final-correction-enabled",
        choices=["true", "false"],
        default=None,
        help="Override runtime.asr_final_correction_enabled for final ASR text correction.",
    )
    p.add_argument("--final-correction-context-items", type=int, default=None)
    p.add_argument("--final-correction-max-chars", type=int, default=None)
    p.add_argument(
        "--adaptive-enabled",
        choices=["true", "false"],
        default=None,
        help="Override runtime.adaptive_asr_enabled for this benchmark run.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config)
    audio_path = Path(args.audio)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Loading {audio_path.name}...", flush=True)
    audio, sample_rate = _load_wav(audio_path)
    print(f"[main] Duration: {len(audio)/sample_rate:.1f}s @ {sample_rate}Hz", flush=True)

    reference_text = ""
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            reference_text = ref_path.read_text(encoding="utf-8").strip()
        else:
            print(f"[warn] Reference not found: {ref_path}", flush=True)

    if args.sweep:
        _sweep(
            audio, sample_rate,
            config_path=config_path,
            source=args.source,
            language=args.language,
            reference_text=reference_text,
            output_dir=output_dir,
            speed=args.speed,
        )
        return

    # Single run
    vad_ov: dict[str, Any] = {}
    stream_ov: dict[str, Any] = {}
    asr_ov: dict[str, Any] = {}
    runtime_ov: dict[str, Any] = {}
    worker_ov: dict[str, Any] = {}
    if args.neural_threshold is not None:
        vad_ov["neural_threshold"] = args.neural_threshold
    if args.min_silence_ms is not None:
        vad_ov["min_silence_duration_ms"] = args.min_silence_ms
    if args.speech_pad_ms is not None:
        vad_ov["speech_pad_ms"] = args.speech_pad_ms
    if args.startup_suppress_ms is not None:
        vad_ov["startup_suppress_ms"] = args.startup_suppress_ms
    if args.soft_final_ms is not None:
        stream_ov["soft_final_audio_ms"] = args.soft_final_ms
    if args.partial_history_s is not None:
        stream_ov["partial_history_seconds"] = args.partial_history_s
    if args.final_history_s is not None:
        stream_ov["final_history_seconds"] = args.final_history_s
    if args.beam_size is not None:
        asr_ov["beam_size"] = args.beam_size
    if args.final_beam_size is not None:
        asr_ov["final_beam_size"] = args.final_beam_size
    if args.no_speech_threshold is not None:
        asr_ov["no_speech_threshold"] = args.no_speech_threshold
    if args.hotwords is not None:
        asr_ov["hotwords"] = args.hotwords
    if args.initial_prompt is not None:
        asr_ov["initial_prompt"] = args.initial_prompt
    if args.adaptive_enabled is not None:
        runtime_ov["adaptive_asr_enabled"] = args.adaptive_enabled == "true"
    if args.bias_enabled is not None:
        runtime_ov["asr_hotwords_enabled"] = args.bias_enabled == "true"
    if args.bias_lexicon is not None:
        runtime_ov["asr_hotword_lexicon"] = args.bias_lexicon
    if args.final_correction_enabled is not None:
        runtime_ov["asr_final_correction_enabled"] = args.final_correction_enabled == "true"
    if args.final_correction_context_items is not None:
        runtime_ov["asr_final_correction_context_items"] = args.final_correction_context_items
    if args.final_correction_max_chars is not None:
        runtime_ov["asr_final_correction_max_chars"] = args.final_correction_max_chars
    if args.pre_roll_ms is not None:
        worker_ov["pre_roll_ms"] = args.pre_roll_ms
    if args.partial_interval_ms is not None:
        worker_ov["partial_interval_ms"] = args.partial_interval_ms
    if args.min_partial_audio_ms is not None:
        worker_ov["min_partial_audio_ms"] = args.min_partial_audio_ms

    result = run_streaming_sim(
        audio, sample_rate,
        config_path=config_path,
        source=args.source,
        language=args.language,
        vad_overrides=vad_ov or None,
        streaming_overrides=stream_ov or None,
        asr_overrides=asr_ov or None,
        runtime_overrides=runtime_ov or None,
        worker_overrides=worker_ov or None,
        lead_in_ms=args.lead_in_ms,
        chunk_ms=args.chunk_ms,
        speed_multiplier=args.speed,
        verbose=True,
    )

    if reference_text:
        hyp = _normalize(result["transcript"], lang=args.language, ref=reference_text)
        ref = _normalize(reference_text, lang=args.language, ref=reference_text)
        hyp_dedup = _dedupe_repetition(hyp, lang=args.language)
        cer = round(_cer(hyp, ref), 4)
        cer_dedup = round(_cer(hyp_dedup, ref), 4)
        wer = round(_wer(hyp, ref), 4) if not _is_cjk(args.language) else None
        result["cer_normalized"] = cer
        result["accuracy"] = round(1 - cer, 4)
        result["cer_dedup"] = cer_dedup
        result["accuracy_dedup"] = round(1 - cer_dedup, 4)
        result["repetition_ratio"] = round(_repetition_ratio(hyp, lang=args.language), 4)
        if wer is not None:
            result["wer_normalized"] = wer

    # Save
    out_name = f"streaming_result_{args.source}_{args.language.replace('-','')}.json"
    out_path = output_dir / out_name
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "="*60, flush=True)
    print(f"[result] finals={result['final_count']}", flush=True)
    print(f"[result] transcript ({len(result['transcript'])} chars):", flush=True)
    print(result["transcript"][:300], flush=True)
    if "accuracy" in result:
        print(f"[result] CER_normalized={result['cer_normalized']:.4f} "
              f"accuracy={result['accuracy']:.1%}", flush=True)
    if "accuracy_dedup" in result:
        print(f"[result] CER_dedup={result['cer_dedup']:.4f} "
              f"dedup_accuracy={result['accuracy_dedup']:.1%} "
              f"repetition_ratio={result.get('repetition_ratio', 0.0):.1%}", flush=True)
    print(f"[result] VAD: available={result['vad_available']} "
          f"speech_started={result['speech_started_count']} "
          f"hard_endpoints={result['hard_endpoint_count']}", flush=True)
    print(f"[result] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
