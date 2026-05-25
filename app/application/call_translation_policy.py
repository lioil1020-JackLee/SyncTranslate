from __future__ import annotations

from dataclasses import dataclass

from app.infra.config.schema import AppConfig, AudioRouteConfig, CallTranslationConfig


SYNC_VIRTUAL_AUDIO = "synctranslate_virtual_audio"


@dataclass(frozen=True, slots=True)
class CallTranslationPolicy:
    routing_mode: str
    listen_remote_original: bool
    listen_remote_translation: bool
    output_local_original: bool
    output_local_translation: bool

    @property
    def uses_virtual_audio(self) -> bool:
        return self.routing_mode == SYNC_VIRTUAL_AUDIO

    @property
    def local_channel_output_mode(self) -> str:
        if self.listen_remote_translation:
            return "tts"
        if self.listen_remote_original:
            return "passthrough"
        return "subtitle_only"

    @property
    def remote_channel_output_mode(self) -> str:
        if self.output_local_translation:
            return "tts"
        if self.output_local_original:
            return "passthrough"
        return "subtitle_only"


def normalize_routing_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == SYNC_VIRTUAL_AUDIO:
        return SYNC_VIRTUAL_AUDIO
    return SYNC_VIRTUAL_AUDIO


def resolve_call_translation_policy(config: AppConfig | AudioRouteConfig) -> CallTranslationPolicy:
    audio = config.audio if isinstance(config, AppConfig) else config
    if isinstance(config, AppConfig) and str(getattr(config.runtime, "session_mode", "meeting") or "meeting") == "meeting":
        return CallTranslationPolicy(
            routing_mode=normalize_routing_mode(getattr(audio, "routing_mode", SYNC_VIRTUAL_AUDIO)),
            listen_remote_original=False,
            listen_remote_translation=False,
            output_local_original=False,
            output_local_translation=False,
        )
    if isinstance(config, AppConfig) and str(getattr(config.runtime, "session_mode", "meeting") or "meeting") == "dialogue":
        return CallTranslationPolicy(
            routing_mode=normalize_routing_mode(getattr(audio, "routing_mode", SYNC_VIRTUAL_AUDIO)),
            listen_remote_original=config.dialogue.remote_to_local.output_policy == "direct_passthrough",
            listen_remote_translation=config.dialogue.remote_to_local.output_policy == "translated_tts",
            output_local_original=config.dialogue.local_to_remote.output_policy == "direct_passthrough",
            output_local_translation=config.dialogue.local_to_remote.output_policy == "translated_tts",
        )
    toggles = getattr(audio, "call_translation", CallTranslationConfig())
    return CallTranslationPolicy(
        routing_mode=normalize_routing_mode(getattr(audio, "routing_mode", SYNC_VIRTUAL_AUDIO)),
        listen_remote_original=bool(getattr(toggles, "listen_remote_original", True)),
        listen_remote_translation=bool(getattr(toggles, "listen_remote_translation", True)),
        output_local_original=bool(getattr(toggles, "output_local_original", False)),
        output_local_translation=bool(getattr(toggles, "output_local_translation", True)),
    )


__all__ = [
    "SYNC_VIRTUAL_AUDIO",
    "CallTranslationPolicy",
    "normalize_routing_mode",
    "resolve_call_translation_policy",
]
