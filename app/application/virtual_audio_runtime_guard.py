from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.application.call_translation_policy import SYNC_VIRTUAL_AUDIO, resolve_call_translation_policy
from app.infra.audio.virtual_devices import detect_virtual_audio_install
from app.infra.config.schema import AppConfig


@dataclass(frozen=True, slots=True)
class VirtualAudioRuntimeGuardResult:
    effective_routing_mode: str
    blocked: bool
    reason: str
    warnings: tuple[str, ...]


def ensure_virtual_audio_runtime_ready(config: AppConfig) -> VirtualAudioRuntimeGuardResult:
    policy = resolve_call_translation_policy(config)
    if policy.routing_mode != SYNC_VIRTUAL_AUDIO:
        return VirtualAudioRuntimeGuardResult(
            effective_routing_mode=policy.routing_mode,
            blocked=False,
            reason="",
            warnings=(),
        )

    warnings: list[str] = []
    virtual_cfg = config.audio.virtual_audio

    bridge_path = str(getattr(virtual_cfg, "bridge_path", "") or "").strip()
    if bridge_path and not Path(bridge_path).exists():
        warnings.append(f"bridge_path_missing:{bridge_path}")

    if not bool(getattr(virtual_cfg, "require_driver", True)):
        return VirtualAudioRuntimeGuardResult(
            effective_routing_mode=SYNC_VIRTUAL_AUDIO,
            blocked=False,
            reason="",
            warnings=tuple(warnings),
        )

    virtual_status = detect_virtual_audio_install()
    missing: list[str] = []
    if not virtual_status.speaker_available:
        missing.append(str(getattr(virtual_cfg, "speaker_name", "") or "SyncTranslate Virtual Speaker"))
    if not virtual_status.microphone_available:
        missing.append(str(getattr(virtual_cfg, "microphone_name", "") or "SyncTranslate Virtual Microphone"))

    if missing:
        reason = "virtual_audio_driver_unavailable:" + "|".join(missing)
        return VirtualAudioRuntimeGuardResult(
            effective_routing_mode=SYNC_VIRTUAL_AUDIO,
            blocked=True,
            reason=reason,
            warnings=tuple(warnings),
        )

    return VirtualAudioRuntimeGuardResult(
        effective_routing_mode=SYNC_VIRTUAL_AUDIO,
        blocked=False,
        reason="",
        warnings=tuple(warnings),
    )


__all__ = ["VirtualAudioRuntimeGuardResult", "ensure_virtual_audio_runtime_ready"]