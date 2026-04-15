from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class SpeakerProfile:
    label: str
    centroid: np.ndarray
    turns: int = 1
    last_seen_ms: int = 0


class OnlineSpeakerDiarizer:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_audio_ms: int = 900,
        max_speakers: int = 3,
        similarity_threshold: float = 0.82,
    ) -> None:
        self._enabled = bool(enabled)
        self._min_audio_ms = max(300, int(min_audio_ms))
        self._max_speakers = max(1, int(max_speakers))
        self._similarity_threshold = max(0.2, min(0.99, float(similarity_threshold)))
        self._profiles: list[SpeakerProfile] = []
        self._last_label = ""
        self._pending_switch_label = ""
        self._pending_switch_count = 0
        self._pending_new_count = 0
        self._switch_margin = 0.09
        self._switch_patience = 2
        self._new_speaker_margin = 0.14
        self._new_speaker_patience = 2
        self._sticky_similarity_floor = max(0.56, self._similarity_threshold - 0.18)
        self._reuse_similarity_floor = max(0.52, self._similarity_threshold - 0.22)
        self._uncertain_similarity_floor = max(0.5, self._similarity_threshold - self._new_speaker_margin + 0.02)

    def reset(self) -> None:
        self._profiles.clear()
        self._last_label = ""
        self._pending_switch_label = ""
        self._pending_switch_count = 0
        self._pending_new_count = 0

    def assign(self, *, audio: np.ndarray, sample_rate: int, now_ms: int) -> str:
        if not self._enabled or sample_rate <= 0:
            return ""
        feature = self._extract_embedding(audio=audio, sample_rate=sample_rate)
        if feature is None:
            return ""
        if not self._profiles:
            profile = self._create_profile(feature, now_ms=now_ms)
            self._last_label = profile.label
            return profile.label

        provisional_hold = False
        best_profile: SpeakerProfile | None = None
        best_similarity = -1.0
        for profile in self._profiles:
            similarity = self._profile_similarity(feature, profile)
            if similarity > best_similarity:
                best_similarity = similarity
                best_profile = profile

        last_profile = self._find_profile(self._last_label)
        last_similarity = self._profile_similarity(feature, last_profile) if last_profile is not None else -1.0

        if self._should_stick_to_last(
            last_profile=last_profile,
            last_similarity=last_similarity,
            best_profile=best_profile,
            best_similarity=best_similarity,
            now_ms=now_ms,
        ):
            assert last_profile is not None
            self._update_profile(last_profile, feature, now_ms=now_ms)
            self._clear_pending_candidates()
            self._last_label = last_profile.label
            return last_profile.label

        if best_profile is not None and best_similarity >= self._similarity_threshold:
            if last_profile is not None and best_profile.label != last_profile.label:
                if not self._confirm_switch(best_profile.label):
                    provisional_hold = True
                    fallback = self._fallback_profile(last_profile=last_profile, last_similarity=last_similarity)
                    if fallback is not None:
                        self._last_label = fallback.label
                        return fallback.label
                else:
                    self._pending_new_count = 0
            else:
                self._clear_pending_candidates()
            self._update_profile(best_profile, feature, now_ms=now_ms)
            self._last_label = best_profile.label
            return best_profile.label

        if (
            len(self._profiles) < self._max_speakers
            and self._should_consider_new_speaker(best_similarity=best_similarity, last_similarity=last_similarity)
        ):
            self._pending_new_count += 1
            self._pending_switch_label = ""
            self._pending_switch_count = 0
            provisional_hold = True
            if self._pending_new_count >= self._new_speaker_patience:
                profile = self._create_profile(feature, now_ms=now_ms)
                self._pending_new_count = 0
                self._last_label = profile.label
                return profile.label
        else:
            self._pending_new_count = 0

        fallback = self._fallback_profile(
            last_profile=last_profile,
            last_similarity=last_similarity,
            best_profile=best_profile,
            best_similarity=best_similarity,
        )
        if fallback is None:
            self._clear_pending_candidates()
            return ""
        if not provisional_hold:
            self._update_profile(fallback, feature, now_ms=now_ms)
        self._last_label = fallback.label
        return fallback.label

    def _create_profile(self, feature: np.ndarray, *, now_ms: int) -> SpeakerProfile:
        index = len(self._profiles)
        label = f"Speaker {chr(ord('A') + index)}"
        profile = SpeakerProfile(label=label, centroid=feature.copy(), turns=1, last_seen_ms=now_ms)
        self._profiles.append(profile)
        return profile

    def _find_profile(self, label: str) -> SpeakerProfile | None:
        if not label:
            return None
        for profile in self._profiles:
            if profile.label == label:
                return profile
        return None

    def _profile_similarity(self, feature: np.ndarray, profile: SpeakerProfile | None) -> float:
        if profile is None:
            return -1.0
        similarity = self._cosine_similarity(feature, profile.centroid)
        if self._pitch_gap_is_too_large(feature, profile.centroid):
            similarity -= 0.25
        return similarity

    def _should_stick_to_last(
        self,
        *,
        last_profile: SpeakerProfile | None,
        last_similarity: float,
        best_profile: SpeakerProfile | None,
        best_similarity: float,
        now_ms: int,
    ) -> bool:
        if last_profile is None:
            return False
        if last_similarity >= self._similarity_threshold:
            return True
        if last_similarity < self._sticky_similarity_floor:
            return False
        if best_profile is None:
            return last_similarity >= self._uncertain_similarity_floor
        if best_profile.label == last_profile.label:
            return last_similarity >= self._uncertain_similarity_floor
        if best_similarity - last_similarity < self._switch_margin:
            return True
        recent_gap_ms = max(0, int(now_ms - last_profile.last_seen_ms))
        return (
            recent_gap_ms <= 4500
            and last_similarity >= self._reuse_similarity_floor
            and (best_similarity - last_similarity) < (self._switch_margin + 0.03)
        )

    def _confirm_switch(self, label: str) -> bool:
        self._pending_new_count = 0
        if self._pending_switch_label == label:
            self._pending_switch_count += 1
        else:
            self._pending_switch_label = label
            self._pending_switch_count = 1
        if self._pending_switch_count >= self._switch_patience:
            self._pending_switch_label = ""
            self._pending_switch_count = 0
            return True
        return False

    def _should_consider_new_speaker(self, *, best_similarity: float, last_similarity: float) -> bool:
        return (
            best_similarity < (self._similarity_threshold - self._new_speaker_margin)
            and last_similarity < self._uncertain_similarity_floor
        )

    def _fallback_profile(
        self,
        *,
        last_profile: SpeakerProfile | None,
        last_similarity: float,
        best_profile: SpeakerProfile | None = None,
        best_similarity: float = -1.0,
    ) -> SpeakerProfile | None:
        if last_profile is not None and last_similarity >= self._reuse_similarity_floor:
            return last_profile
        if best_profile is not None and best_similarity >= self._reuse_similarity_floor:
            return best_profile
        return last_profile or best_profile

    def _clear_pending_candidates(self) -> None:
        self._pending_switch_label = ""
        self._pending_switch_count = 0
        self._pending_new_count = 0

    @staticmethod
    def _update_profile(profile: SpeakerProfile, feature: np.ndarray, *, now_ms: int) -> None:
        turns = max(1, int(profile.turns))
        weight_old = min(0.88, turns / (turns + 1.5))
        profile.centroid = (profile.centroid * weight_old) + (feature * (1.0 - weight_old))
        norm = float(np.linalg.norm(profile.centroid))
        if norm > 1e-6:
            profile.centroid = profile.centroid / norm
        profile.turns = turns + 1
        profile.last_seen_ms = now_ms

    def _extract_embedding(self, *, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        if mono.size <= 1:
            return None
        audio_ms = int(round(mono.shape[0] * 1000.0 / float(sample_rate)))
        if audio_ms < self._min_audio_ms:
            return None

        centered = mono - float(np.mean(mono))
        rms = float(np.sqrt(np.mean(np.square(centered)))) if centered.size else 0.0
        if rms < 0.008:
            return None

        target_rate = min(16000, int(sample_rate))
        reduced = self._resample_linear(centered, sample_rate=sample_rate, target_rate=target_rate)
        voiced = self._trim_to_voiced(reduced, sample_rate=target_rate)
        if voiced.size < int(target_rate * 0.25):
            return None

        pitch_hz, harmonicity = self._estimate_pitch(voiced, sample_rate=target_rate)
        centroid, rolloff = self._spectral_features(voiced, sample_rate=target_rate)
        zcr = self._zero_crossing_rate(voiced)
        band_low, band_mid, band_high = self._band_energy_ratios(voiced, sample_rate=target_rate)

        pitch_norm = min(1.0, max(0.0, (pitch_hz - 70.0) / 260.0)) if pitch_hz > 0 else 0.0
        centroid_norm = min(1.0, centroid / max(1.0, target_rate / 2.0))
        rolloff_norm = min(1.0, rolloff / max(1.0, target_rate / 2.0))
        rms_norm = min(1.0, rms / 0.2)

        vector = np.array(
            [
                pitch_norm,
                pitch_norm,
                pitch_norm,
                min(1.0, harmonicity),
                min(1.0, harmonicity),
                centroid_norm,
                rolloff_norm,
                min(1.0, zcr * 6.0),
                band_low,
                band_mid,
                band_high,
                rms_norm,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    @staticmethod
    def _resample_linear(audio: np.ndarray, *, sample_rate: int, target_rate: int) -> np.ndarray:
        if sample_rate <= 0 or target_rate <= 0 or sample_rate == target_rate or audio.size <= 1:
            return audio.astype(np.float32, copy=False)
        src_len = int(audio.shape[0])
        dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
        src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)

    @staticmethod
    def _trim_to_voiced(audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        frame = max(128, int(sample_rate * 0.03))
        hop = max(64, int(sample_rate * 0.015))
        if audio.size <= frame:
            return audio
        rms_values: list[float] = []
        ranges: list[tuple[int, int]] = []
        for start in range(0, max(1, audio.size - frame), hop):
            end = min(audio.size, start + frame)
            chunk = audio[start:end]
            rms_values.append(float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0)
            ranges.append((start, end))
        if not rms_values:
            return audio
        rms_array = np.asarray(rms_values, dtype=np.float32)
        threshold = max(0.008, float(np.percentile(rms_array, 55)) * 0.85)
        voiced_ranges = [ranges[idx] for idx, value in enumerate(rms_array) if value >= threshold]
        if not voiced_ranges:
            return audio
        start = voiced_ranges[0][0]
        end = voiced_ranges[-1][1]
        return audio[start:end]

    @staticmethod
    def _estimate_pitch(audio: np.ndarray, *, sample_rate: int) -> tuple[float, float]:
        if audio.size <= 1:
            return 0.0, 0.0
        min_hz = 80.0
        max_hz = 320.0
        min_lag = max(1, int(sample_rate / max_hz))
        max_lag = max(min_lag + 1, int(sample_rate / min_hz))
        window = audio[: min(audio.size, int(sample_rate * 1.2))]
        if window.size <= max_lag + 1:
            return 0.0, 0.0
        window = window - float(np.mean(window))
        corr = np.correlate(window, window, mode="full")[window.size - 1 :]
        if corr.size <= max_lag:
            return 0.0, 0.0
        peak_slice = corr[min_lag:max_lag]
        if peak_slice.size == 0:
            return 0.0, 0.0
        peak_index = int(np.argmax(peak_slice)) + min_lag
        peak_value = float(corr[peak_index])
        zero_lag = max(1e-6, float(corr[0]))
        harmonicity = max(0.0, min(1.0, peak_value / zero_lag))
        if harmonicity < 0.18:
            return 0.0, harmonicity
        return float(sample_rate) / float(peak_index), harmonicity

    @staticmethod
    def _spectral_features(audio: np.ndarray, *, sample_rate: int) -> tuple[float, float]:
        if audio.size <= 1:
            return 0.0, 0.0
        window = np.hanning(audio.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
        total = float(np.sum(spectrum))
        if total <= 1e-6:
            return 0.0, 0.0
        centroid = float(np.sum(freqs * spectrum) / total)
        cumulative = np.cumsum(spectrum)
        threshold = cumulative[-1] * 0.85
        rolloff_idx = int(np.searchsorted(cumulative, threshold))
        rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
        return centroid, rolloff

    @staticmethod
    def _zero_crossing_rate(audio: np.ndarray) -> float:
        if audio.size <= 1:
            return 0.0
        signs = np.signbit(audio)
        crossings = np.count_nonzero(signs[1:] != signs[:-1])
        return float(crossings) / float(audio.size - 1)

    @staticmethod
    def _band_energy_ratios(audio: np.ndarray, *, sample_rate: int) -> tuple[float, float, float]:
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
        power = np.square(spectrum, dtype=np.float32)
        total = float(np.sum(power))
        if total <= 1e-6:
            return 0.0, 0.0, 0.0

        def _band(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]) / total)

        return _band(80.0, 500.0), _band(500.0, 1500.0), _band(1500.0, 4000.0)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-6:
            return -1.0
        return float(np.dot(left, right) / denom)

    @staticmethod
    def _pitch_gap_is_too_large(left: np.ndarray, right: np.ndarray) -> bool:
        left_pitch = float(left[0]) if left.size else 0.0
        right_pitch = float(right[0]) if right.size else 0.0
        left_harmonicity = float(left[3]) if left.size > 3 else 0.0
        right_harmonicity = float(right[3]) if right.size > 3 else 0.0
        if min(left_harmonicity, right_harmonicity) < 0.22:
            return False
        return abs(left_pitch - right_pitch) >= 0.12


__all__ = ["OnlineSpeakerDiarizer", "SpeakerProfile"]
