from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FinalRescuePolicy:
    enabled: bool = True
    max_attempts: int = 1
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.65
    max_compression_ratio: float = 2.4
    min_chars: int = 4


def compact_len(text: str) -> int:
    return len("".join(str(text or "").split()))


def confidence_failure_reasons(
    result: object,
    text: str,
    *,
    audio_ms: int,
    policy: FinalRescuePolicy,
) -> list[str]:
    if not policy.enabled or int(policy.max_attempts) <= 0:
        return []
    value = str(text or "").strip()
    compact = compact_len(value)
    reasons: list[str] = []

    avg_logprob = getattr(result, "avg_logprob", None)
    if avg_logprob is not None and float(avg_logprob) < float(policy.min_avg_logprob):
        reasons.append("low_logprob")

    no_speech_prob = getattr(result, "max_no_speech_prob", None)
    if no_speech_prob is not None and float(no_speech_prob) > float(policy.max_no_speech_prob):
        reasons.append("high_no_speech")

    compression_ratio = getattr(result, "max_compression_ratio", None)
    if compression_ratio is not None and float(compression_ratio) > float(policy.max_compression_ratio):
        reasons.append("high_compression")

    if not value and int(audio_ms) >= 700:
        reasons.append("empty_text")
    elif compact < int(policy.min_chars) and int(audio_ms) >= 900:
        reasons.append("too_short_for_audio")

    return reasons


def candidate_score(result: object, text: str) -> float:
    value = str(text or "").strip()
    if not value:
        return -9999.0
    score = 0.0
    avg_logprob = getattr(result, "avg_logprob", None)
    if avg_logprob is not None:
        score += float(avg_logprob) * 2.0
    no_speech_prob = getattr(result, "max_no_speech_prob", None)
    if no_speech_prob is not None:
        score -= float(no_speech_prob) * 1.5
    compression_ratio = getattr(result, "max_compression_ratio", None)
    if compression_ratio is not None:
        score -= max(0.0, float(compression_ratio) - 1.0) * 0.5
    score += min(1.5, compact_len(value) / 20.0)
    return score


def choose_better_candidate(current: object, candidate: object) -> object:
    current_text = str(getattr(current, "text", "") or "")
    candidate_text = str(getattr(candidate, "text", "") or "")
    if not candidate_text.strip():
        return current
    if not current_text.strip():
        return candidate
    current_score = candidate_score(current, current_text)
    candidate_score_value = candidate_score(candidate, candidate_text)
    current_len = compact_len(current_text)
    candidate_len = compact_len(candidate_text)
    if current_len >= 8 and candidate_len + max(4, int(current_len * 0.25)) < current_len:
        if candidate_score_value <= current_score + 1.0:
            return current
    if candidate_score_value > current_score + 0.15:
        return candidate
    if candidate_len >= current_len + 4 and candidate_score_value >= current_score - 0.05:
        return candidate
    return current


__all__ = [
    "FinalRescuePolicy",
    "candidate_score",
    "choose_better_candidate",
    "compact_len",
    "confidence_failure_reasons",
]
