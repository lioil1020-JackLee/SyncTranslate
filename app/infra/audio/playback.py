from __future__ import annotations

import platform
import time
from threading import Event, Lock, Thread

import numpy as np
import sounddevice as sd

try:
    import soundcard as sc
except Exception:
    sc = None

from app.infra.audio.device_registry import (
    canonical_device_name,
    device_tokens,
    hostapi_name_by_index,
    list_indexed_devices,
    normalize_device_text,
    parse_device_selector,
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
            self._stop_current_stream()
            errors: list[str] = []
            try:
                if self._try_play_via_soundcard(
                    audio=audio,
                    sample_rate=requested_sample_rate,
                    output_device_name=output_device_name,
                ):
                    return
            except Exception as exc:
                # If soundcard backend fails, continue to sounddevice fallback candidates.
                errors.append(str(exc))

            tried_indices: set[int] = set()
            selected_candidates = self._find_output_devices(output_device_name)
            if self._try_play_on_candidates(
                candidates=selected_candidates,
                audio=audio,
                requested_sample_rate=requested_sample_rate,
                errors=errors,
                tried_indices=tried_indices,
            ):
                return

            # If selector includes a hostapi and that path fails, retry by device name only.
            # This recovers from stale/unsupported hostapi choices (e.g. DirectSound vs WASAPI).
            canonical_name = canonical_device_name(output_device_name)
            if canonical_name and canonical_name != output_device_name:
                fallback_candidates = self._find_output_devices(canonical_name)
                fallback_candidates = [pair for pair in fallback_candidates if pair[0] not in tried_indices]
                if self._try_play_on_candidates(
                    candidates=fallback_candidates,
                    audio=audio,
                    requested_sample_rate=requested_sample_rate,
                    errors=errors,
                    tried_indices=tried_indices,
                ):
                    return

            if not errors:
                raise ValueError(f"Output device not found: {output_device_name}")
            raise ValueError(
                "Unable to play TTS audio on the selected output device. "
                + " | ".join(errors[:6])
            )

    def stop(self) -> None:
        with self._play_lock:
            self._stop_current_stream()

    def _stop_current_stream(self) -> None:
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
        if self._can_play_via_soundcard(output_device_name):
            return float(requested_sample_rate)
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

    def _try_play_on_candidates(
        self,
        *,
        candidates: list[tuple[int, dict[str, object]]],
        audio: np.ndarray,
        requested_sample_rate: float,
        errors: list[str],
        tried_indices: set[int],
    ) -> bool:
        for device_index, device_info in candidates:
            tried_indices.add(device_index)
            default_sample_rate = float(device_info["default_samplerate"])
            max_output_channels = int(device_info["max_output_channels"])
            prefer_stereo = self._should_prefer_stereo_output(str(device_info["name"]))
            prefer_device_rate = self._should_prefer_device_sample_rate(str(device_info["name"]))
            for playback_audio, channels in self._candidate_audio_variants(
                audio=audio,
                max_output_channels=max_output_channels,
                prefer_stereo=prefer_stereo,
            ):
                try:
                    resolved_sample_rate = self._resolve_supported_output_sample_rate(
                        device_index=device_index,
                        channels=channels,
                        requested_sample_rate=requested_sample_rate,
                        default_sample_rate=default_sample_rate,
                        prefer_device_rate=prefer_device_rate,
                    )
                    playback_audio = self._resample_audio_if_needed(
                        audio=playback_audio,
                        source_sample_rate=requested_sample_rate,
                        target_sample_rate=resolved_sample_rate,
                    )
                    playback_audio = self._apply_volume(playback_audio)
                    hostapi_name = hostapi_name_by_index(int(device_info.get("hostapi", -1))).strip().lower()
                    timeout_sec = max(3.0, float(playback_audio.shape[0]) / max(1.0, resolved_sample_rate) + 2.0)
                    if hostapi_name == "windows wdm-ks":
                        self._play_via_callback_stream(
                            device_index=device_index,
                            samplerate=resolved_sample_rate,
                            channels=channels,
                            audio=playback_audio,
                            timeout_sec=timeout_sec,
                        )
                    else:
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
                            self._write_stream_blocking_with_timeout(
                                stream=stream,
                                audio=playback_audio,
                                timeout_sec=timeout_sec,
                            )
                            stream.stop()
                        finally:
                            with self._state_lock:
                                if self._stream is stream:
                                    self._stream = None
                            stream.close()
                    self._last_play_device = str(device_info["name"])
                    return True
                except Exception as exc:
                    errors.append(f"{device_info['name']} [idx={device_index}, {channels}ch]: {exc}")
        return False

    def _try_play_via_soundcard(self, *, audio: np.ndarray, sample_rate: float, output_device_name: str) -> bool:
        if platform.system().lower() != "windows" or sc is None:
            return False
        if self._should_avoid_soundcard_backend(output_device_name):
            return False

        speaker = self._find_soundcard_speaker(output_device_name)
        if speaker is None:
            return False

        requested_rate = int(round(sample_rate))
        max_channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        candidate_rates: list[int] = []
        for rate in (requested_rate, 48000, 44100, 24000):
            if rate > 0 and rate not in candidate_rates:
                candidate_rates.append(rate)

        last_error: Exception | None = None
        for channels in (2, 1):
            if channels < max_channels:
                continue
            prepared_audio = self._prepare_soundcard_audio(audio=audio, channels=channels)
            for rate in candidate_rates:
                try:
                    playback_audio = self._resample_audio_if_needed(
                        audio=prepared_audio,
                        source_sample_rate=sample_rate,
                        target_sample_rate=float(rate),
                    )
                    playback_audio = self._apply_volume(playback_audio)
                    with speaker.player(samplerate=rate, channels=channels, blocksize=1024) as player:
                        player.play(playback_audio.astype(np.float32, copy=False))
                    self._last_play_device = speaker.name
                    return True
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            raise ValueError(f"Soundcard playback failed for {speaker.name}: {last_error}") from last_error
        return False

    def _can_play_via_soundcard(self, output_device_name: str) -> bool:
        if self._should_avoid_soundcard_backend(output_device_name):
            return False
        return self._find_soundcard_speaker(output_device_name) is not None

    @staticmethod
    def _should_avoid_soundcard_backend(output_device_name: str) -> bool:
        normalized = normalize_device_text(canonical_device_name(output_device_name))
        return any(token in normalized for token in ("voicemeeter", "vb audio", "virtual"))

    @staticmethod
    def _find_soundcard_speaker(output_device_name: str):
        if sc is None:
            return None

        target_name = canonical_device_name(output_device_name)
        normalized_target = normalize_device_text(target_name)
        target_tokens = device_tokens(target_name)
        ranked: list[tuple[int, int, str, int, object]] = []
        for speaker_index, speaker in enumerate(sc.all_speakers()):
            name = str(getattr(speaker, "name", "") or "")
            normalized_name = normalize_device_text(name)
            name_tokens = device_tokens(name)
            score = 0
            if name == target_name:
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
            if score > 0:
                extra_token_penalty = max(0, len(name_tokens) - len(target_tokens))
                ranked.append((-score, extra_token_penalty, normalized_name, speaker_index, speaker))

        ranked.sort()
        return ranked[0][4] if ranked else None

    @staticmethod
    def _prepare_soundcard_audio(*, audio: np.ndarray, channels: int) -> np.ndarray:
        base = audio.astype(np.float32, copy=False)
        if base.ndim == 1:
            mono = base.reshape(-1, 1)
        elif base.ndim == 2:
            mono = base[:, :1]
        else:
            raise ValueError(f"Unsupported audio shape for playback: {audio.shape}")

        if channels <= 1:
            return mono
        return np.repeat(mono, channels, axis=1)

    @staticmethod
    def _find_output_devices(device_name: str) -> list[tuple[int, dict[str, object]]]:
        hostapi_name, requested_name = parse_device_selector(device_name)
        devices = list_indexed_devices()
        preferred_hostapi = preferred_hostapi_index_for_platform()
        normalized_target = normalize_device_text(requested_name)
        target_tokens = device_tokens(requested_name)
        target_looks_virtual = any(token in normalized_target for token in ("voicemeeter", "vb audio", "virtual", "cable"))
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
                resolved_hostapi_name = str(sd.query_hostapis()[hostapi].get("name", ""))
                hostapi_matches = resolved_hostapi_name == hostapi_name
                # For virtual devices, tolerate stale DirectSound selection and prefer WASAPI/KS.
                if not hostapi_matches:
                    if target_looks_virtual:
                        lowered = resolved_hostapi_name.strip().lower()
                        if lowered not in ("windows wasapi", "windows wdm-ks"):
                            continue
                    else:
                        continue

                if target_looks_virtual:
                    lowered = resolved_hostapi_name.strip().lower()
                    if lowered == "windows wasapi":
                        hostapi_rank = -2
                    elif lowered == "windows wdm-ks":
                        hostapi_rank = -1
                    elif hostapi_matches:
                        hostapi_rank = 0
                    else:
                        hostapi_rank = 1
                else:
                    hostapi_rank = 0 if hostapi_matches else 1
            else:
                hostapi_rank = 0 if hostapi == preferred_hostapi else 1
            extra_token_penalty = max(0, len(name_tokens) - len(target_tokens))
            ranked.append((hostapi_rank, -score, extra_token_penalty, idx, item))

        ranked.sort()
        return [(idx, item) for _, _, _, idx, item in ranked]

    @staticmethod
    def _write_stream_blocking_with_timeout(
        *,
        stream: sd.OutputStream,
        audio: np.ndarray,
        timeout_sec: float,
    ) -> None:
        contiguous_audio = np.ascontiguousarray(audio, dtype=np.float32)
        failure: list[Exception] = []

        def _writer() -> None:
            try:
                stream.write(contiguous_audio)
            except Exception as exc:
                failure.append(exc)

        writer = Thread(target=_writer, daemon=True)
        writer.start()
        writer.join(timeout=max(0.5, timeout_sec))
        if writer.is_alive():
            try:
                stream.abort()
            except Exception:
                pass
            raise TimeoutError(f"Output stream write timed out after {timeout_sec:.1f}s")
        if failure:
            raise failure[0]

    def _play_via_callback_stream(
        self,
        *,
        device_index: int,
        samplerate: float,
        channels: int,
        audio: np.ndarray,
        timeout_sec: float,
    ) -> None:
        contiguous_audio = np.ascontiguousarray(audio, dtype=np.float32)
        if contiguous_audio.ndim == 1:
            contiguous_audio = contiguous_audio.reshape(-1, 1)
        if contiguous_audio.shape[1] != channels:
            if channels == 1:
                contiguous_audio = contiguous_audio[:, :1]
            else:
                contiguous_audio = np.repeat(contiguous_audio[:, :1], channels, axis=1)

        done = Event()
        cursor = 0

        def _callback(outdata: np.ndarray, frames: int, _time_info, status) -> None:
            nonlocal cursor
            if status:
                # status may contain underflow flags even when playback proceeds.
                pass
            end = min(cursor + frames, contiguous_audio.shape[0])
            chunk = contiguous_audio[cursor:end]
            outdata[: len(chunk)] = chunk
            if len(chunk) < frames:
                outdata[len(chunk) :] = 0
                done.set()
                raise sd.CallbackStop()
            cursor = end

        stream = sd.OutputStream(
            device=device_index,
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            callback=_callback,
        )
        with self._state_lock:
            self._stream = stream
        try:
            stream.start()
            deadline = time.monotonic() + max(0.5, timeout_sec)
            while not done.is_set() and time.monotonic() < deadline:
                sd.sleep(10)
            if not done.is_set():
                try:
                    stream.abort()
                except Exception:
                    pass
                raise TimeoutError(f"Output callback playback timed out after {timeout_sec:.1f}s")
            stream.stop()
        finally:
            with self._state_lock:
                if self._stream is stream:
                    self._stream = None
            stream.close()

    @staticmethod
    def _resolve_supported_output_sample_rate(
        *,
        device_index: int,
        channels: int,
        requested_sample_rate: float,
        default_sample_rate: float,
        prefer_device_rate: bool = False,
    ) -> float:
        candidate_rates: list[float] = []
        if prefer_device_rate:
            for rate in (default_sample_rate, 48000.0, 44100.0, requested_sample_rate):
                if rate > 0 and rate not in candidate_rates:
                    candidate_rates.append(rate)
        elif requested_sample_rate > 0:
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
        prefer_stereo: bool = False,
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

        variants: list[tuple[np.ndarray, int]] = []
        if max_output_channels >= 2:
            stereo = np.repeat(mono, 2, axis=1)
            if prefer_stereo:
                variants.append((stereo, 2))
                variants.append((mono, 1))
            else:
                variants.append((mono, 1))
                variants.append((stereo, 2))
        else:
            variants.append((mono, 1))
        return variants

    @staticmethod
    def _should_prefer_stereo_output(device_name: str) -> bool:
        normalized = normalize_device_text(device_name)
        return any(token in normalized for token in ("voicemeeter", "vb-audio", "virtual"))

    @staticmethod
    def _should_prefer_device_sample_rate(device_name: str) -> bool:
        normalized = normalize_device_text(device_name)
        return any(token in normalized for token in ("voicemeeter", "vb-audio", "virtual"))

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


__all__ = ["AudioPlayback"]
