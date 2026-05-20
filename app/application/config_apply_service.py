from __future__ import annotations

from app.infra.config.schema import AppConfig, AudioRouteConfig


class ConfigApplyService:
    def __init__(
        self,
        *,
        meeting_capture,
        local_capture,
        speaker_playback,
        meeting_playback,
        audio_routing_page,
        live_caption_page,
        local_ai_page,
        device_volume_controller: object | None = None,
    ) -> None:
        del meeting_capture, local_capture, speaker_playback, meeting_playback, device_volume_controller
        self._audio_routing_page = audio_routing_page
        self._live_caption_page = live_caption_page
        self._local_ai_page = local_ai_page

    def sync_ui_to_config(self, config: AppConfig) -> None:
        config.audio = self._audio_routing_page.selected_audio_routes()
        self._live_caption_page.update_config(config)
        self._local_ai_page.update_config(config)

    def sync_live_caption_to_config(self, config: AppConfig) -> None:
        self._live_caption_page.update_config(config)

    def apply_audio_route_levels(self, config: AppConfig | AudioRouteConfig) -> None:
        del config

    def apply_audio_route_levels_from_ui(self, config: AppConfig | None = None) -> None:
        if config is not None:
            config.audio = self._audio_routing_page.selected_audio_routes()
