from __future__ import annotations

from app.infra.audio.capture import AudioCapture
from app.infra.audio.device_volume_controller import SystemDeviceVolumeController
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
        device_volume_controller: SystemDeviceVolumeController | None = None,
    ) -> None:
        self._meeting_capture = meeting_capture
        self._local_capture = local_capture
        self._speaker_playback = speaker_playback
        self._meeting_playback = meeting_playback
        self._audio_routing_page = audio_routing_page
        self._live_caption_page = live_caption_page
        self._local_ai_page = local_ai_page
        self._device_volume_controller = device_volume_controller or SystemDeviceVolumeController()

    def sync_ui_to_config(self, config: AppConfig) -> None:
        config.audio = self._audio_routing_page.selected_audio_routes()
        config.runtime.passthrough_gain = 1.0
        config.runtime.tts_gain = 1.0
        self._live_caption_page.update_config(config)
        self._local_ai_page.update_config(config)
        self.apply_audio_route_levels(config)

    def sync_live_caption_to_config(self, config: AppConfig) -> None:
        config.runtime.passthrough_gain = 1.0
        config.runtime.tts_gain = 1.0
        self._live_caption_page.update_config(config)
        self.apply_audio_route_levels(config)

    def apply_audio_route_levels(self, config: AppConfig | AudioRouteConfig) -> None:
        if isinstance(config, AppConfig):
            audio = config.audio
        else:
            audio = config
        meeting_in_gain = max(0.0, min(1.0, float(getattr(audio, "meeting_in_gain", 1.0) or 1.0)))
        microphone_in_gain = max(0.0, min(1.0, float(getattr(audio, "microphone_in_gain", 1.0) or 1.0)))
        speaker_out_volume = max(0.0, min(1.0, float(getattr(audio, "speaker_out_volume", 1.0) or 1.0)))
        meeting_out_volume = max(0.0, min(1.0, float(getattr(audio, "meeting_out_volume", 1.0) or 1.0)))
        self._meeting_capture.set_gain(1.0)
        self._local_capture.set_gain(1.0)
        self._speaker_playback.set_volume(1.0)
        self._meeting_playback.set_volume(1.0)
        audio.meeting_in_gain = meeting_in_gain
        audio.microphone_in_gain = microphone_in_gain
        audio.speaker_out_volume = speaker_out_volume
        audio.meeting_out_volume = meeting_out_volume
        self._device_volume_controller.apply_audio_route_config(audio)

    def apply_audio_route_levels_from_ui(self, config: AppConfig | None = None) -> None:
        audio = self._audio_routing_page.selected_audio_routes()
        if config is not None:
            config.audio = audio
            audio = config.audio
        self.apply_audio_route_levels(audio)
