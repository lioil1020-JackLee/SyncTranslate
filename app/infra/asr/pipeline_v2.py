from __future__ import annotations

from dataclasses import dataclass

from app.infra.asr.backend_resolution import resolve_backend_for_language
from app.infra.asr.backend_v2 import BackendDescriptor
from app.infra.asr.endpointing_v2 import EndpointingDescriptor, build_endpointing_descriptor
from app.infra.config.schema import AppConfig


@dataclass(slots=True)
class AsrV2PipelineSpec:
    pipeline_name: str
    execution_mode: str
    partial_backend: BackendDescriptor
    final_backend: BackendDescriptor
    endpointing: EndpointingDescriptor
    correction_enabled: bool
    diarization_enabled: bool


def resolve_requested_asr_language(config: AppConfig, source: str) -> str:
    runtime = config.runtime
    explicit = (
        str(getattr(runtime, "remote_asr_language", "") or "").strip()
        if source == "remote"
        else str(getattr(runtime, "local_asr_language", "") or "").strip()
    )
    if explicit:
        return explicit
    if source == "remote":
        return str(getattr(config.language, "meeting_source", "") or "").strip()
    return str(getattr(config.language, "local_source", "") or "").strip()


def build_v2_pipeline_spec(config: AppConfig) -> AsrV2PipelineSpec:
    local_language = resolve_requested_asr_language(config, "local")
    resolution = resolve_backend_for_language(local_language)
    endpointing = build_endpointing_descriptor(
        str(getattr(config.runtime, "asr_v2_endpointing", "neural_endpoint")),
        config.asr_channels.local.vad,
        resolved_backend_name=resolution.backend_name,
    )
    backend_name = resolution.backend_name if not resolution.disabled else "disabled"
    notes = resolution.reason
    return AsrV2PipelineSpec(
        pipeline_name="asr_v2",
        execution_mode="native_v2",
        partial_backend=BackendDescriptor(
            name=f"{backend_name}:partial",
            mode="native_v2",
            streaming=True,
            notes=notes or "Low-latency partial decoding backend.",
        ),
        final_backend=BackendDescriptor(
            name=f"{backend_name}:final",
            mode="native_v2",
            streaming=False,
            notes=notes or "High-quality final decoding backend.",
        ),
        endpointing=endpointing,
        correction_enabled=bool(getattr(config.runtime, "asr_final_correction_enabled", False)),
        diarization_enabled=bool(getattr(config.runtime, "speaker_diarization_enabled", False)),
    )


__all__ = ["AsrV2PipelineSpec", "build_v2_pipeline_spec", "resolve_requested_asr_language"]
