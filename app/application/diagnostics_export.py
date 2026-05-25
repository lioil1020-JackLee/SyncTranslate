from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from app.infra.audio.virtual_bridge_probe import probe_virtual_audio_bridge
from app.infra.audio.bridge_protocol import BRIDGE_PROTOCOL_VERSION
from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.config.schema import AppConfig, translation_enabled_for_source
from app.domain.version import build_metadata
from app.version import app_version


def _latency_histogram(values: list[float]) -> str:
    bins = {
        "0_500": 0,
        "500_1000": 0,
        "1000_2000": 0,
        "2000_4000": 0,
        "4000_plus": 0,
    }
    for raw in values:
        value = max(0.0, float(raw))
        if value < 500.0:
            bins["0_500"] += 1
        elif value < 1000.0:
            bins["500_1000"] += 1
        elif value < 2000.0:
            bins["1000_2000"] += 1
        elif value < 4000.0:
            bins["2000_4000"] += 1
        else:
            bins["4000_plus"] += 1
    return (
        f"0-500={bins['0_500']} "
        f"500-1000={bins['500_1000']} "
        f"1000-2000={bins['1000_2000']} "
        f"2000-4000={bins['2000_4000']} "
        f">=4000={bins['4000_plus']}"
    )


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
    virtual_audio_status = detect_virtual_audio_install()
    metadata = build_metadata(
        config_schema_version=int(getattr(config.runtime, "config_schema_version", 0) or 0),
        runtime_mode=str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
    )
    bridge_probe = (
        probe_virtual_audio_bridge(config.audio.virtual_audio.bridge_path)
        if bool(config.audio.virtual_audio.bridge_enabled)
        else None
    )

    overflow = {}
    latency_summary: list[str] = []
    latency_samples_ms: list[float] = []
    if router_stats:
        overflow = router_stats.get("translation_overflow") or {}
        for entry in list(router_stats.get("latency") or [])[:8]:
            latency_summary.append(str(entry))
            for key in (
                "first_asr_partial_ms",
                "first_display_partial_ms",
                "speech_end_to_asr_final_ms",
                "asr_final_to_llm_final_ms",
                "tts_enqueue_to_playback_start_ms",
            ):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    latency_samples_ms.append(float(value))
    asr_observation_summary: list[str] = []
    for label, source in (("meeting", "remote"), ("local", "local")):
        asr_stats = ((router_stats or {}).get("asr") or {}).get(source) or {}
        post = asr_stats.get("postprocessor") or {}
        final_post = post.get("final") or {}
        endpointing = asr_stats.get("endpointing") or {}
        endpoint_signal = asr_stats.get("endpoint_signal") or {}
        asr_observation_summary.append(
            f"{label}: "
            f"q={int(asr_stats.get('queue_size', 0) or 0)} "
            f"p={int(asr_stats.get('partial_count', 0) or 0)} "
            f"f={int(asr_stats.get('final_count', 0) or 0)} "
            f"pause_ms={int(endpoint_signal.get('pause_ms', 0) or 0)} "
            f"qmax={int(asr_stats.get('queue_maxsize', 0) or 0)} "
            f"drop={int(asr_stats.get('dropped_chunks', 0) or 0)} "
            f"model={str(asr_stats.get('configured_model', '') or '-')} "
            f"fallback={str(asr_stats.get('fallback_model', '') or '-')} "
            f"profile={str(asr_stats.get('endpoint_profile', '') or '-')} "
            f"beam={int(asr_stats.get('configured_beam_size', 0) or 0)}/"
            f"{int(asr_stats.get('configured_final_beam_size', 0) or 0)} "
            f"partial_ms={int(asr_stats.get('configured_partial_interval_ms', 0) or 0)} "
            f"final_hist={int(asr_stats.get('configured_final_history_seconds', 0) or 0)} "
            f"vad={int(asr_stats.get('configured_vad_min_speech_ms', 0) or 0)}/"
            f"{int(asr_stats.get('configured_vad_min_silence_ms', 0) or 0)}/"
            f"{int(asr_stats.get('configured_vad_speech_pad_ms', 0) or 0)} "
            f"enh={bool(asr_stats.get('frontend_enhancement_enabled', False))} "
            f"fp={bool(asr_stats.get('final_priority_active', False))} "
            f"rescue={int(asr_stats.get('final_rescue_count', 0) or 0)}/"
            f"{int(asr_stats.get('final_fallback_count', 0) or 0)} "
            f"speech_started={int(endpointing.get('speech_started_count', 0) or 0)} "
            f"soft={int(endpointing.get('soft_endpoint_count', 0) or 0)} "
            f"hard={int(endpointing.get('hard_endpoint_count', 0) or 0)} "
            f"rej={int(final_post.get('rejected_count', 0) or 0)} "
            f"last_rej={str(final_post.get('last_rejection_reason', '') or '-')}"
        )

    content = "\n".join(
        [
            "SyncTranslate Diagnostics",
            f"timestamp: {datetime.now().isoformat()}",
            f"app_version: {metadata.app_version}",
            f"git_commit: {metadata.git_commit}",
            f"build_timestamp: {metadata.build_timestamp}",
            f"packaged: {metadata.packaged}",
            f"config_path: {config_path}",
            f"session_mode: {str(getattr(config.runtime, 'session_mode', 'meeting') or 'meeting')}",
            f"legacy_direction_mode: {config.direction.mode}",
            f"audio_routing_mode: {config.audio.routing_mode}",
            f"virtual_speaker_name: {config.audio.virtual_audio.speaker_name}",
            f"virtual_microphone_name: {config.audio.virtual_audio.microphone_name}",
            f"virtual_bridge_enabled: {config.audio.virtual_audio.bridge_enabled}",
            f"virtual_bridge_ready: {bool(getattr(bridge_probe, 'ready', False))}",
            f"virtual_bridge_connected: {bool(getattr(bridge_probe, 'connected', False))}",
            f"virtual_bridge_error: {str(getattr(bridge_probe, 'error', '') or '')}",
            f"virtual_bridge_shared_memory_name: {str(getattr(bridge_probe, 'shared_memory_name', '') or '')}",
            f"virtual_bridge_event_name: {str(getattr(bridge_probe, 'event_name', '') or '')}",
            f"virtual_bridge_heartbeat_ok: {bool(getattr(bridge_probe, 'heartbeat_ok', False))}",
            f"virtual_bridge_heartbeat_roundtrip_ms: {float(getattr(bridge_probe, 'heartbeat_roundtrip_ms', 0.0) or 0.0):.2f}",
            f"virtual_bridge_loopback_ok: {bool(getattr(bridge_probe, 'loopback_ok', False))}",
            f"virtual_bridge_loopback_latency_ms: {float(getattr(bridge_probe, 'loopback_latency_ms', 0.0) or 0.0):.2f}",
            f"virtual_bridge_remote_input_running: {bool(getattr(bridge_probe, 'remote_input_running', False))}",
            f"virtual_bridge_remote_input_frames: {int(getattr(bridge_probe, 'remote_input_frames', 0) or 0)}",
            f"virtual_bridge_remote_input_buffered_frames: {int(getattr(bridge_probe, 'remote_input_buffered_frames', 0) or 0)}",
            f"virtual_bridge_remote_input_capacity_frames: {int(getattr(bridge_probe, 'remote_input_buffer_capacity_frames', 0) or 0)}",
            f"virtual_bridge_virtual_mic_frames: {int(getattr(bridge_probe, 'virtual_microphone_frames', 0) or 0)}",
            f"virtual_bridge_virtual_mic_buffered_frames: {int(getattr(bridge_probe, 'virtual_microphone_buffered_frames', 0) or 0)}",
            f"virtual_bridge_virtual_mic_dropped_frames: {int(getattr(bridge_probe, 'virtual_microphone_dropped_frames', 0) or 0)}",
            f"virtual_bridge_virtual_mic_capacity_frames: {int(getattr(bridge_probe, 'virtual_microphone_buffer_capacity_frames', 0) or 0)}",
            f"virtual_driver_installed: {virtual_audio_status.installed}",
            f"virtual_driver_speaker_available: {virtual_audio_status.speaker_available}",
            f"virtual_driver_microphone_available: {virtual_audio_status.microphone_available}",
            f"virtual_driver_detected_speaker: {virtual_audio_status.speaker_name}",
            f"virtual_driver_detected_microphone: {virtual_audio_status.microphone_name}",
            f"virtual_driver_detected_speaker_index: {virtual_audio_status.speaker_index}",
            f"virtual_driver_detected_microphone_index: {virtual_audio_status.microphone_index}",
            f"remote_translation_target: {config.language.meeting_target}",
            f"local_translation_target: {config.language.local_target}",
            f"meeting_in: {routes.meeting_in}",
            f"microphone_in: {routes.microphone_in}",
            f"speaker_out: {routes.speaker_out}",
            f"meeting_out: {routes.meeting_out}",
            f"meeting_source_type: {str(getattr(config.meeting, 'audio_source', 'system_input') or 'system_input')}",
            f"meeting_asr_language: {str(getattr(config.meeting, 'asr_language', 'zh-TW') or 'zh-TW')}",
            f"dialogue_remote_asr_language: {str(getattr(config.dialogue, 'remote_asr_language', 'en') or 'en')}",
            f"dialogue_local_asr_language: {str(getattr(config.dialogue, 'local_asr_language', 'zh-TW') or 'zh-TW')}",
            f"direct_passthrough_remote_to_local: {config.dialogue.remote_to_local.output_policy == 'direct_passthrough'}",
            f"direct_passthrough_local_to_remote: {config.dialogue.local_to_remote.output_policy == 'direct_passthrough'}",
            f"capture_kind: {'output_loopback' if str(getattr(config.meeting, 'audio_source', 'system_input') or '') == 'system_output_loopback' else 'input'}",
            f"capture_sample_rate: {int(float(((router_stats or {}).get('capture') or {}).get('remote', {}).get('sample_rate', 0) or 0))}",
            f"capture_channels: {int(((router_stats or {}).get('capture') or {}).get('remote', {}).get('channels', 0) or 0)}",
            "asr_input_sample_rate: 16000",
            "asr_input_channels: 1",
            f"bridge_required: {str(getattr(config.runtime, 'session_mode', 'meeting') or 'meeting') == 'dialogue' and bool(config.audio.virtual_audio.bridge_enabled)}",
            f"bridge_connected: {bool(getattr(bridge_probe, 'connected', False))}",
            f"driver_available: {bool(virtual_audio_status.speaker_available and virtual_audio_status.microphone_available)}",
            f"last_route_policy_reason: {'meeting_monitor_no_virtual_audio' if str(getattr(config.runtime, 'session_mode', 'meeting') or 'meeting') == 'meeting' else 'dialogue_virtual_audio_routes'}",
            f"asr_model: {config.asr.model}",
            f"asr_device: {config.asr.device}",
            f"asr_compute_type: {config.asr.compute_type}",
            f"llm_backend: {config.llm.backend}",
            f"llm_model_path: {config.llm.runtime.model_path}",
            f"llm_model: {config.llm.model}",
            f"llm_ctx_size: {config.llm.runtime.ctx_size}",
            f"llm_gpu_layers: {config.llm.runtime.gpu_layers}",
            f"llm_threads: {config.llm.runtime.threads}",
            f"remote_translation_enabled: {translation_enabled_for_source(config.runtime, 'remote')}",
            f"local_translation_enabled: {translation_enabled_for_source(config.runtime, 'local')}",
            "asr_language_mode: fixed",
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
            f"asr_accuracy_mode: {str(getattr(config.runtime, 'asr_accuracy_mode', 'balanced') or 'balanced')}",
            f"asr_final_rescue_enabled: {bool(getattr(config.runtime, 'asr_final_rescue_enabled', True))}",
            f"asr_chinese_fallback_enabled: {bool(getattr(config.runtime, 'asr_chinese_fallback_enabled', True))}",
            f"translation_overflow_local: {overflow.get('local', 0)}",
            f"translation_overflow_remote: {overflow.get('remote', 0)}",
            f"latency_histogram_ms: {_latency_histogram(latency_samples_ms)}",
            "asr_observation:",
            *asr_observation_summary,
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
    virtual_audio_status = detect_virtual_audio_install()
    bridge_probe = (
        probe_virtual_audio_bridge(config.audio.virtual_audio.bridge_path)
        if bool(config.audio.virtual_audio.bridge_enabled)
        else None
    )
    report = {
        "timestamp": now.isoformat(),
        "versions": {
            "app": app_version(),
            "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
            "driver": "",
            "msi": "",
        },
        "config_path": config_path,
        "session_mode": str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
        "legacy_direction_mode": config.direction.mode,
        "selected_devices": {
            "meeting_in": routes.meeting_in,
            "microphone_in": routes.microphone_in,
            "speaker_out": routes.speaker_out,
            "meeting_out": routes.meeting_out,
        },
        "audio_routing": {
            "mode": config.audio.routing_mode,
            "virtual_speaker_name": config.audio.virtual_audio.speaker_name,
            "virtual_microphone_name": config.audio.virtual_audio.microphone_name,
            "bridge_enabled": config.audio.virtual_audio.bridge_enabled,
            "bridge_path": config.audio.virtual_audio.bridge_path,
            "bridge_ready": bool(getattr(bridge_probe, "ready", False)),
            "bridge_connected": bool(getattr(bridge_probe, "connected", False)),
            "bridge_error": str(getattr(bridge_probe, "error", "") or ""),
            "bridge_shared_memory_name": str(getattr(bridge_probe, "shared_memory_name", "") or ""),
            "bridge_event_name": str(getattr(bridge_probe, "event_name", "") or ""),
            "bridge_heartbeat_ok": bool(getattr(bridge_probe, "heartbeat_ok", False)),
            "bridge_heartbeat_roundtrip_ms": float(getattr(bridge_probe, "heartbeat_roundtrip_ms", 0.0) or 0.0),
            "bridge_loopback_ok": bool(getattr(bridge_probe, "loopback_ok", False)),
            "bridge_loopback_latency_ms": float(getattr(bridge_probe, "loopback_latency_ms", 0.0) or 0.0),
            "bridge_remote_input_running": bool(getattr(bridge_probe, "remote_input_running", False)),
            "bridge_remote_input_frames": int(getattr(bridge_probe, "remote_input_frames", 0) or 0),
            "bridge_remote_input_buffered_frames": int(getattr(bridge_probe, "remote_input_buffered_frames", 0) or 0),
            "bridge_remote_input_capacity_frames": int(getattr(bridge_probe, "remote_input_buffer_capacity_frames", 0) or 0),
            "bridge_virtual_mic_frames": int(getattr(bridge_probe, "virtual_microphone_frames", 0) or 0),
            "bridge_virtual_mic_buffered_frames": int(getattr(bridge_probe, "virtual_microphone_buffered_frames", 0) or 0),
            "bridge_virtual_mic_dropped_frames": int(getattr(bridge_probe, "virtual_microphone_dropped_frames", 0) or 0),
            "bridge_virtual_mic_capacity_frames": int(getattr(bridge_probe, "virtual_microphone_buffer_capacity_frames", 0) or 0),
            "require_driver": config.audio.virtual_audio.require_driver,
            "driver_installed": virtual_audio_status.installed,
            "driver_speaker_available": virtual_audio_status.speaker_available,
            "driver_microphone_available": virtual_audio_status.microphone_available,
            "driver_detected_speaker": virtual_audio_status.speaker_name,
            "driver_detected_microphone": virtual_audio_status.microphone_name,
            "driver_detected_speaker_index": virtual_audio_status.speaker_index,
            "driver_detected_microphone_index": virtual_audio_status.microphone_index,
            "driver_render_endpoint_count": len(virtual_audio_status.render_endpoints),
            "driver_capture_endpoint_count": len(virtual_audio_status.capture_endpoints),
            "call_translation": {
                "listen_remote_original": config.audio.call_translation.listen_remote_original,
                "listen_remote_translation": config.audio.call_translation.listen_remote_translation,
                "output_local_original": config.audio.call_translation.output_local_original,
                "output_local_translation": config.audio.call_translation.output_local_translation,
            },
        },
        "backend": {
            "llm_backend": config.llm.backend,
            "llm_model": config.llm.model,
            "llm_model_path": config.llm.runtime.model_path,
            "llm_ctx_size": config.llm.runtime.ctx_size,
            "llm_gpu_layers": config.llm.runtime.gpu_layers,
            "llm_threads": config.llm.runtime.threads,
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
                "session_mode": str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
                "meeting_audio_source": str(getattr(config.meeting, "audio_source", "system_input") or "system_input"),
                "capture_kind": "output_loopback"
                if str(getattr(config.meeting, "audio_source", "system_input") or "") == "system_output_loopback"
                else "input",
                "asr_input_sample_rate": 16000,
                "asr_input_channels": 1,
                "direct_passthrough_active_remote": config.dialogue.remote_to_local.output_policy == "direct_passthrough",
                "direct_passthrough_active_local": config.dialogue.local_to_remote.output_policy == "direct_passthrough",
                "chunk_ms": config.runtime.chunk_ms,
            "remote_translation_enabled": translation_enabled_for_source(config.runtime, "remote"),
            "local_translation_enabled": translation_enabled_for_source(config.runtime, "local"),
            "asr_language_mode": "fixed",
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
            "asr_accuracy_mode": str(getattr(config.runtime, "asr_accuracy_mode", "balanced") or "balanced"),
            "asr_final_rescue_enabled": bool(getattr(config.runtime, "asr_final_rescue_enabled", True)),
            "asr_chinese_fallback_enabled": bool(getattr(config.runtime, "asr_chinese_fallback_enabled", True)),
        },
        "stats": payload.get("stats_before_stop", {}),
        "translation_overflow": (payload.get("stats_before_stop") or {}).get("translation_overflow", {}),
        "recent_latency": list((payload.get("stats_before_stop") or {}).get("latency") or [])[:16],
        "latency_histogram_ms": _latency_histogram(
            [
                float(item[key])
                for item in list((payload.get("stats_before_stop") or {}).get("latency") or [])[:64]
                for key in (
                    "first_asr_partial_ms",
                    "first_display_partial_ms",
                    "speech_end_to_asr_final_ms",
                    "asr_final_to_llm_final_ms",
                    "tts_enqueue_to_playback_start_ms",
                )
                if isinstance(item.get(key), (int, float))
            ]
        ),
        "session_meta": payload.get("session_meta", {}),
        "recent_errors": recent_errors[-50:],
        "config_snapshot": config.to_dict(),
    }
    output_path = report_dir / f"session_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
