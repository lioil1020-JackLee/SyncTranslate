from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from app.infra.config.schema import AppConfig, translation_enabled_for_source


def export_runtime_diagnostics(
    *,
    config_path: str,
    config: AppConfig,
    routes,
    runtime_stats_text: str,
    recent_errors: list[str],
    router_stats: dict[str, object] | None = None,
) -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"diagnostics_{now}.txt")

    overflow = {}
    latency_summary: list[str] = []
    if router_stats:
        overflow = router_stats.get("translation_overflow") or {}
        for entry in list(router_stats.get("latency") or [])[:8]:
            latency_summary.append(str(entry))

    content = "\n".join(
        [
            "SyncTranslate Diagnostics",
            f"timestamp: {datetime.now().isoformat()}",
            f"config_path: {config_path}",
            f"mode: {config.direction.mode}",
            f"remote_translation_target: {config.language.meeting_target}",
            f"local_translation_target: {config.language.local_target}",
            f"meeting_in: {routes.meeting_in}",
            f"microphone_in: {routes.microphone_in}",
            f"speaker_out: {routes.speaker_out}",
            f"meeting_out: {routes.meeting_out}",
            f"asr_model: {config.asr.model}",
            f"asr_device: {config.asr.device}",
            f"asr_compute_type: {config.asr.compute_type}",
            f"llm_backend: {config.llm.backend}",
            f"llm_url: {config.llm.base_url}",
            f"llm_model: {config.llm.model}",
            f"remote_translation_enabled: {translation_enabled_for_source(config.runtime, 'remote')}",
            f"local_translation_enabled: {translation_enabled_for_source(config.runtime, 'local')}",
            "asr_language_mode: auto",
            f"tts_output_mode: {str(getattr(config.runtime, 'tts_output_mode', 'subtitle_only') or 'subtitle_only')}",
            f"meeting_tts_engine: {config.meeting_tts.engine}",
            f"meeting_tts_model: {config.meeting_tts.model_path}",
            f"meeting_tts_voice: {config.meeting_tts.voice_name}",
            f"local_tts_engine: {config.local_tts.engine}",
            f"local_tts_model: {config.local_tts.model_path}",
            f"local_tts_voice: {config.local_tts.voice_name}",
            f"sample_rate: {config.runtime.sample_rate}",
            f"max_pipeline_latency_ms: {config.runtime.max_pipeline_latency_ms}",
            f"display_partial_strategy: {config.runtime.display_partial_strategy}",
            f"llm_queue_maxsize_local: {config.runtime.llm_queue_maxsize_local}",
            f"llm_queue_maxsize_remote: {config.runtime.llm_queue_maxsize_remote}",
            f"translation_overflow_local: {overflow.get('local', 0)}",
            f"translation_overflow_remote: {overflow.get('remote', 0)}",
            "recent_latency_entries:",
            *latency_summary,
            "runtime_stats:",
            runtime_stats_text,
            "recent_errors:",
            *recent_errors[-30:],
        ]
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def export_session_report(
    *,
    config_path: str,
    config: AppConfig,
    routes,
    payload: dict[str, object],
    recent_errors: list[str],
) -> Path:
    report_dir = Path("logs") / "session_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    report = {
        "timestamp": now.isoformat(),
        "config_path": config_path,
        "mode": config.direction.mode,
        "selected_devices": {
            "meeting_in": routes.meeting_in,
            "microphone_in": routes.microphone_in,
            "speaker_out": routes.speaker_out,
            "meeting_out": routes.meeting_out,
        },
        "backend": {
            "llm_backend": config.llm.backend,
            "llm_model": config.llm.model,
            "llm_url": config.llm.base_url,
            "asr_model": config.asr.model,
            "asr_device": config.asr.device,
            "meeting_tts": {
                "engine": config.meeting_tts.engine,
                "voice": config.meeting_tts.voice_name,
            },
            "local_tts": {
                "engine": config.local_tts.engine,
                "voice": config.local_tts.voice_name,
            },
        },
        "runtime": {
            "sample_rate": config.runtime.sample_rate,
            "chunk_ms": config.runtime.chunk_ms,
            "remote_translation_enabled": translation_enabled_for_source(config.runtime, "remote"),
            "local_translation_enabled": translation_enabled_for_source(config.runtime, "local"),
            "asr_language_mode": "auto",
            "tts_output_mode": str(getattr(config.runtime, "tts_output_mode", "subtitle_only") or "subtitle_only"),
            "max_pipeline_latency_ms": config.runtime.max_pipeline_latency_ms,
            "display_partial_strategy": config.runtime.display_partial_strategy,
            "asr_queue_maxsize_local": config.runtime.asr_queue_maxsize_local,
            "asr_queue_maxsize_remote": config.runtime.asr_queue_maxsize_remote,
            "llm_queue_maxsize_local": config.runtime.llm_queue_maxsize_local,
            "llm_queue_maxsize_remote": config.runtime.llm_queue_maxsize_remote,
            "tts_queue_maxsize_local": config.runtime.tts_queue_maxsize_local,
            "tts_queue_maxsize_remote": config.runtime.tts_queue_maxsize_remote,
            "asr_queue_maxsize": config.runtime.asr_queue_maxsize,
            "llm_queue_maxsize": config.runtime.llm_queue_maxsize,
            "tts_queue_maxsize": config.runtime.tts_queue_maxsize,
        },
        "stats": payload.get("stats_before_stop", {}),
        "translation_overflow": (payload.get("stats_before_stop") or {}).get("translation_overflow", {}),
        "recent_latency": list((payload.get("stats_before_stop") or {}).get("latency") or [])[:16],
        "session_meta": payload.get("session_meta", {}),
        "recent_errors": recent_errors[-50:],
        "config_snapshot": config.to_dict(),
    }
    output_path = report_dir / f"session_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
