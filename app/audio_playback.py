from __future__ import annotations

from threading import Lock

import numpy as np
import sounddevice as sd

from app.audio_device_selection import (
    device_tokens,
    normalize_device_text,
    parse_device_selector,
    list_indexed_devices,
    preferred_hostapi_index_for_platform,
)


class AudioPlayback:
    def __init__(self) -> None:
        self._last_play_device: str = ""
        self._volume = 1.0
        self._state_lock = Lock()
        self._play_lock = Lock()
        self._stream: sd.OutputStream | None = None

    def play(self, audio: np.ndarray, sample_rate: int, output_device_name: str, *, blocking: bool = False) -> None:
        if audio.size == 0 or not output_device_name:
            return
        requested_sample_rate = float(sample_rate)
        with self._play_lock:
            self.stop()
            errors: list[str] = []
            for device_index, device_info in self._find_output_devices(output_device_name):
                default_sample_rate = float(device_info["default_samplerate"])
                max_output_channels = int(device_info["max_output_channels"])
                for playback_audio, channels in self._candidate_audio_variants(
                    audio=audio,
                    max_output_channels=max_output_channels,
                ):
                    try:
                        resolved_sample_rate = self._resolve_supported_output_sample_rate(
                            device_index=device_index,
                            channels=channels,
                            requested_sample_rate=requested_sample_rate,
                            default_sample_rate=default_sample_rate,
                        )
                        playback_audio = self._resample_audio_if_needed(
                            audio=playback_audio,
                            source_sample_rate=requested_sample_rate,
                            target_sample_rate=resolved_sample_rate,
                        )
                        playback_audio = self._apply_volume(playback_audio)
                        stream = sd.OutputStream(
                            device=device_index,
                            samplerate=resolved_sample_rate,
                            channels=channels,
                            dtype="float32",
                        )
                        with self._state_lock:
                            self._stream = stream
                        try:
                            stream.start()
                            stream.write(np.ascontiguousarray(playback_audio, dtype=np.float32))
                            stream.stop()
                        finally:
                            with self._state_lock:
                                if self._stream is stream:
                                    self._stream = None
                            stream.close()
                        self._last_play_device = str(device_info["name"])
                        return
                    except Exception as exc:
                        errors.append(f"{device_info['name']} [idx={device_index}, {channels}ch]: {exc}")

            if not errors:
                raise ValueError(f"Output device not found: {output_device_name}")
            raise ValueError(
                "Unable to play TTS audio on the selected output device or compatible fallback. "
                + " | ".join(errors[:6])
            )

    def stop(self) -> None:
        with self._state_lock:
            stream = self._stream
            self._stream = None
        if not stream:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def set_volume(self, volume: float) -> None:
        with self._state_lock:
            self._volume = max(0.0, float(volume))

    def preview_resolved_sample_rate(self, output_device_name: str, requested_sample_rate: int) -> float:
        requested = float(requested_sample_rate)
        errors: list[str] = []
        for device_index, device_info in self._find_output_devices(output_device_name):
            max_output_channels = int(device_info["max_output_channels"])
            if max_output_channels <= 0:
                continue
            channels = 2 if max_output_channels >= 2 else 1
            default_rate = float(device_info["default_samplerate"])
            try:
                return self._resolve_supported_output_sample_rate(
                    device_index=device_index,
                    channels=channels,
                    requested_sample_rate=requested,
                    default_sample_rate=default_rate,
                )
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError(errors[0])
        raise ValueError(f"Output device not found or unavailable: {output_device_name}")

    @staticmethod
    def _find_output_devices(device_name: str) -> list[tuple[int, dict[str, object]]]:
        hostapi_name, requested_name = parse_device_selector(device_name)
        devices = list_indexed_devices()
        preferred_hostapi = preferred_hostapi_index_for_platform()
        normalized_target = normalize_device_text(requested_name)
        target_tokens = device_tokens(requested_name)
        ranked: list[tuple[int, int, int, int, dict[str, object]]] = []

        for idx, item in devices:
            if int(item["max_output_channels"]) <= 0:
                continue
            name = str(item["name"])
            normalized_name = normalize_device_text(name)
            name_tokens = device_tokens(name)

            score = 0
            if name == requested_name:
                score = 500
            elif normalized_name == normalized_target:
                score = 450
            elif normalized_target and normalized_target in normalized_name:
                score = 350
            elif target_tokens and target_tokens.issubset(name_tokens):
                score = 300 + len(target_tokens)
            elif target_tokens:
                overlap = len(target_tokens & name_tokens)
                if overlap >= max(2, len(target_tokens) - 1):
                    score = 200 + overlap

            if score <= 0:
                continue

            hostapi = int(item.get("hostapi", -1))
            if hostapi_name:
                hostapi_matches = str(sd.query_hostapis()[hostapi].get("name", "")) == hostapi_name
                hostapi_rank = 0 if hostapi_matches else 1
            else:
                hostapi_rank = 0 if hostapi == preferred_hostapi else 1
            extra_token_penalty = max(0, len(name_tokens) - len(target_tokens))
            ranked.append((hostapi_rank, -score, extra_token_penalty, idx, item))

        ranked.sort()
        return [(idx, item) for _, _, _, idx, item in ranked]

    @staticmethod
    def _resolve_supported_output_sample_rate(
        *,
        device_index: int,
        channels: int,
        requested_sample_rate: float,
        default_sample_rate: float,
    ) -> float:
        candidate_rates: list[float] = []
        if requested_sample_rate > 0:
            candidate_rates.append(requested_sample_rate)
        # Prefer common real-time voice sample rates across virtual/physical devices.
        for rate in (48000.0, 44100.0, 32000.0, 24000.0, 22050.0, 16000.0):
            if rate not in candidate_rates:
                candidate_rates.append(rate)
        if default_sample_rate > 0 and default_sample_rate not in candidate_rates:
            candidate_rates.append(default_sample_rate)

        errors: list[str] = []
        for rate in candidate_rates:
            try:
                sd.check_output_settings(
                    device=device_index,
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                )
                return rate
            except Exception as exc:
                errors.append(f"{int(round(rate))}Hz -> {exc}")

        details = "; ".join(errors) if errors else "no valid candidate sample rate"
        raise ValueError(f"Output device does not support requested sample rate(s): {details}")

    @staticmethod
    def _candidate_audio_variants(
        *,
        audio: np.ndarray,
        max_output_channels: int,
    ) -> list[tuple[np.ndarray, int]]:
        if audio.size == 0 or max_output_channels <= 0:
            return []

        base = audio.astype(np.float32, copy=False)
        if base.ndim == 1:
            mono = base.reshape(-1, 1)
        elif base.ndim == 2:
            mono = base[:, :1]
        else:
            raise ValueError(f"Unsupported audio shape for playback: {audio.shape}")

        variants: list[tuple[np.ndarray, int]] = [(mono, 1)]
        if max_output_channels >= 2:
            stereo = np.repeat(mono, 2, axis=1)
            variants.append((stereo, 2))
        return variants

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

    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        with self._state_lock:
            volume = self._volume
        if volume == 1.0:
            return audio.astype(np.float32, copy=False)
        scaled = audio.astype(np.float32, copy=True)
        scaled *= volume
        np.clip(scaled, -1.0, 1.0, out=scaled)
        return scaled
