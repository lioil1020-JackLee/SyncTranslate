from __future__ import annotations

import numpy as np
import sounddevice as sd

from app.audio_device_selection import list_indexed_devices, pick_best_device


class AudioPlayback:
    def __init__(self) -> None:
        self._last_play_device: str = ""

    def play(self, audio: np.ndarray, sample_rate: int, output_device_name: str) -> None:
        if audio.size == 0 or not output_device_name:
            return
        device_index, device_info = self._find_output_device(output_device_name)
        requested_sample_rate = float(sample_rate)
        default_sample_rate = float(device_info["default_samplerate"])
        resolved_sample_rate = self._resolve_supported_output_sample_rate(
            device_index=device_index,
            requested_sample_rate=requested_sample_rate,
            default_sample_rate=default_sample_rate,
        )
        playback_audio = self._resample_audio_if_needed(
            audio=audio,
            source_sample_rate=requested_sample_rate,
            target_sample_rate=resolved_sample_rate,
        )
        sd.play(playback_audio, samplerate=resolved_sample_rate, device=device_index, blocking=False)
        self._last_play_device = output_device_name

    def stop(self) -> None:
        sd.stop()

    @staticmethod
    def _find_output_device(device_name: str) -> tuple[int, dict[str, object]]:
        devices = list_indexed_devices()
        exact_matches = [
            (idx, item)
            for idx, item in devices
            if str(item["name"]) == device_name and int(item["max_output_channels"]) > 0
        ]
        best_exact = pick_best_device(exact_matches)
        if best_exact:
            return best_exact

        partial_matches = [
            (idx, item)
            for idx, item in devices
            if device_name.lower() in str(item["name"]).lower() and int(item["max_output_channels"]) > 0
        ]
        best_partial = pick_best_device(partial_matches)
        if best_partial:
            return best_partial
        raise ValueError(f"Output device not found: {device_name}")

    @staticmethod
    def _resolve_supported_output_sample_rate(
        *,
        device_index: int,
        requested_sample_rate: float,
        default_sample_rate: float,
    ) -> float:
        candidate_rates: list[float] = []
        if requested_sample_rate > 0:
            candidate_rates.append(requested_sample_rate)
        if default_sample_rate > 0 and default_sample_rate not in candidate_rates:
            candidate_rates.append(default_sample_rate)

        errors: list[str] = []
        for rate in candidate_rates:
            try:
                sd.check_output_settings(device=device_index, samplerate=rate)
                return rate
            except Exception as exc:
                errors.append(f"{int(round(rate))}Hz -> {exc}")

        details = "; ".join(errors) if errors else "no valid candidate sample rate"
        raise ValueError(f"Output device does not support requested sample rate(s): {details}")

    @staticmethod
    def _resample_audio_if_needed(
        *,
        audio: np.ndarray,
        source_sample_rate: float,
        target_sample_rate: float,
    ) -> np.ndarray:
        if audio.size == 0:
            return audio

        src_rate = int(round(source_sample_rate))
        dst_rate = int(round(target_sample_rate))
        if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
            return audio.astype(np.float32, copy=False)

        if audio.ndim == 1:
            src = audio.astype(np.float32, copy=False).reshape(-1, 1)
            squeeze = True
        else:
            src = audio.astype(np.float32, copy=False)
            squeeze = False

        src_len = int(src.shape[0])
        if src_len <= 1:
            return audio.astype(np.float32, copy=False)

        dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
        dst = np.empty((dst_len, src.shape[1]), dtype=np.float32)
        for ch in range(src.shape[1]):
            dst[:, ch] = np.interp(dst_x, src_x, src[:, ch]).astype(np.float32, copy=False)

        if squeeze:
            return dst[:, 0]
        return dst
