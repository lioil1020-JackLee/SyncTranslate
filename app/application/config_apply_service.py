from __future__ import annotations

from app.infra.audio.capture import AudioCapture
from app.infra.audio.playback import AudioPlayback
from app.infra.config.schema import AppConfig, AudioRouteConfig


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
        config.direction.mode = "bidirectional"
        self._live_caption_page.update_config(config)
        self._local_ai_page.update_config(config)
        self.apply_audio_route_levels(config)

    def sync_live_caption_to_config(self, config: AppConfig) -> None:
        config.direction.mode = "bidirectional"
        self._live_caption_page.update_config(config)
        self.apply_audio_route_levels(config)

    def apply_audio_route_levels(self, config: AppConfig | AudioRouteConfig) -> None:
        if isinstance(config, AppConfig):
            audio = config.audio
            tts_gain = max(0.0, float(getattr(config.runtime, "tts_gain", 1.4) or 1.4))
        else:
            audio = config
            tts_gain = max(0.0, float(getattr(self._live_caption_page, "selected_tts_gain", lambda: 1.4)()))
        self._meeting_capture.set_gain(1.0)
        self._local_capture.set_gain(1.0)
        self._speaker_playback.set_volume(tts_gain)
        self._meeting_playback.set_volume(tts_gain)
        audio.meeting_in_gain = 1.0
        audio.microphone_in_gain = 1.0
        audio.speaker_out_volume = tts_gain
        audio.meeting_out_volume = tts_gain

    def apply_audio_route_levels_from_ui(self) -> None:
        audio = self._audio_routing_page.selected_audio_routes()
        self.apply_audio_route_levels(audio)
