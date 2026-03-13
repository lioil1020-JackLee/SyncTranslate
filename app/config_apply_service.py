from __future__ import annotations

from app.audio_capture import AudioCapture
from app.audio_playback import AudioPlayback
from app.schemas import AppConfig, AudioRouteConfig


class ConfigApplyService:
    def __init__(
        self,
        *,
        meeting_capture: AudioCapture,
        local_capture: AudioCapture,
        speaker_playback: AudioPlayback,
        meeting_playback: AudioPlayback,
        audio_routing_page,
        live_caption_page,
        local_ai_page,
    ) -> None:
        self._meeting_capture = meeting_capture
        self._local_capture = local_capture
        self._speaker_playback = speaker_playback
        self._meeting_playback = meeting_playback
        self._audio_routing_page = audio_routing_page
        self._live_caption_page = live_caption_page
        self._local_ai_page = local_ai_page

    def sync_ui_to_config(self, config: AppConfig) -> None:
        config.audio = self._audio_routing_page.selected_audio_routes()
        self.apply_audio_route_levels(config.audio)
        config.direction.mode = self._live_caption_page.selected_mode()
        self._live_caption_page.update_config(config)
        self._local_ai_page.update_config(config)

    def apply_audio_route_levels(self, audio: AudioRouteConfig) -> None:
        self._meeting_capture.set_gain(1.0)
        self._local_capture.set_gain(1.0)
        self._speaker_playback.set_volume(1.0)
        self._meeting_playback.set_volume(1.0)
        audio.meeting_in_gain = 1.0
        audio.microphone_in_gain = 1.0
        audio.speaker_out_volume = 1.0
        audio.meeting_out_volume = 1.0

    def apply_audio_route_levels_from_ui(self) -> None:
        audio = self._audio_routing_page.selected_audio_routes()
        self.apply_audio_route_levels(audio)
