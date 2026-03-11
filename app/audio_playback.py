from __future__ import annotations

import numpy as np
import sounddevice as sd


class AudioPlayback:
    def __init__(self) -> None:
        self._last_play_device: str = ""

    def play(self, audio: np.ndarray, sample_rate: int, output_device_name: str) -> None:
        if audio.size == 0 or not output_device_name:
            return
        device_index = self._find_output_device(output_device_name)
        sd.play(audio, samplerate=sample_rate, device=device_index, blocking=False)
        self._last_play_device = output_device_name

    def stop(self) -> None:
        sd.stop()

    @staticmethod
    def _find_output_device(device_name: str) -> int:
        devices = sd.query_devices()
        exact_matches = [
            idx
            for idx, item in enumerate(devices)
            if str(item["name"]) == device_name and int(item["max_output_channels"]) > 0
        ]
        if exact_matches:
            return exact_matches[0]

        partial_matches = [
            idx
            for idx, item in enumerate(devices)
            if device_name.lower() in str(item["name"]).lower() and int(item["max_output_channels"]) > 0
        ]
        if partial_matches:
            return partial_matches[0]
        raise ValueError(f"Output device not found: {device_name}")

