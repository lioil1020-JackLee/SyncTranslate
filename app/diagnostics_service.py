from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from app.schemas import AppConfig


def export_runtime_diagnostics(
    *,
    config_path: str,
    config: AppConfig,
    routes,
    runtime_stats_text: str,
    recent_errors: list[str],
) -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"diagnostics_{now}.txt")
    content = "\n".join(
        [
            "SyncTranslate Diagnostics",
            f"timestamp: {datetime.now().isoformat()}",
            f"config_path: {config_path}",
            f"mode: {config.direction.mode}",
            f"meeting_language: {config.language.meeting_source} -> {config.language.meeting_target}",
            f"local_language: {config.language.local_source} -> {config.language.local_target}",
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
            f"meeting_tts_engine: {config.meeting_tts.engine}",
            f"meeting_tts_model: {config.meeting_tts.model_path}",
            f"meeting_tts_voice: {config.meeting_tts.voice_name}",
            f"local_tts_engine: {config.local_tts.engine}",
            f"local_tts_model: {config.local_tts.model_path}",
            f"local_tts_voice: {config.local_tts.voice_name}",
            f"sample_rate: {config.runtime.sample_rate}",
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
            "asr_queue_maxsize": config.runtime.asr_queue_maxsize,
            "llm_queue_maxsize": config.runtime.llm_queue_maxsize,
            "tts_queue_maxsize": config.runtime.tts_queue_maxsize,
        },
        "stats": payload.get("stats_before_stop", {}),
        "session_meta": payload.get("session_meta", {}),
        "recent_errors": recent_errors[-50:],
        "config_snapshot": config.to_dict(),
    }
    output_path = report_dir / f"session_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
