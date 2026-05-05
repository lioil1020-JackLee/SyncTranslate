"""Hallucination-filter helpers for ASR transcripts.

Contains the constants and stateless functions used to detect and drop both
tail-hallucination transcripts (e.g. spurious "thank you for watching" tokens
generated from silence or low-energy noise) and known non-speech overlays
(watermarks, channel identifiers, etc.).
"""
from __future__ import annotations

import re

from app.domain.unicode_utils import (
    contains_cyrillic_or_greek,
    is_cjk_char,
    is_hangul_char,
    is_japanese_char,
    is_latin_char,
)

_SHORT_TAIL_HALLUCINATION_NORMALIZED = {
    "you",
    "bye",
    "byebye",
    "thankyou",
    "thanks",
    "thankyouall",
    "thankyoueveryone",
    "thankseveryone",
    "thanksall",
    "thankyouforwatching",
    "thanksforwatching",
    "thanksforyourwatching",
    "goodnight",
}

_TAIL_HALLUCINATION_PATTERNS = (
    re.compile(r"^\s*thank(s| you)( everyone| all)?[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thank(s| you)?\s+for\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thanks\s+for\s+your\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*bye(-| )?bye?[.! ]*$", re.IGNORECASE),
)

_CJK_TAIL_HALLUCINATION_PATTERNS = (
    re.compile(r"^[哈呵嘿哇嗚呃嗯啊]{2,}[!！。,.，\s]*$"),
    re.compile(r"^(記得)?按(下|讚).*(訂閱|小鈴鐺)?.*$"),
    re.compile(r"^.*(安妞|按鈕哦|訂閱按鈕|小鈴鐺|頻道嘍|頻道囉|記得按讚).{0,8}$"),
    re.compile(r"^歡迎收看.*下次見[!！。,.，\s]*$"),
)


def looks_like_short_cta_tail(text: str) -> bool:
    """Return True when *text* looks like a CJK call-to-action tail hallucination."""
    value = (text or "").strip()
    if not value:
        return False
    compact = "".join(value.split())
    if len(compact) > 24:
        return False
    return any(pattern.match(value) for pattern in _CJK_TAIL_HALLUCINATION_PATTERNS)


def looks_like_repetitive_loop(text: str) -> bool:
    """Return True when *text* contains a repetitive token loop."""
    tokens = [token for token in text.split() if token]
    if len(tokens) >= 8:
        for size in range(3, min(8, len(tokens) // 2 + 1)):
            chunk = tokens[-size:]
            prev = tokens[-size * 2 : -size]
            if prev == chunk:
                return True
    compact = "".join(text.split())
    if len(compact) < 12:
        return False
    max_span = min(18, len(compact) // 2)
    for span in range(6, max_span + 1):
        suffix = compact[-span:]
        prev = compact[-span * 2 : -span]
        if prev == suffix:
            return True
    return False


def tail_hallucination_drop_reason(
    text: str,
    *,
    audio_ms: int,
    audio_ms_effective: int,
    trailing_silence_ms: int,
    speech_ratio: float,
    mean_rms: float,
    max_rms: float,
) -> str:
    """Return a non-empty reason string when *text* should be dropped as a hallucination.

    Parameters are the raw audio signal statistics for the segment.
    """
    value = (text or "").strip()
    if not value:
        return ""
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    compact = re.sub(r"\s+", " ", value)
    effective_audio_ms = max(0, int(audio_ms), int(audio_ms_effective))
    _trailing_silence_ms = max(0, int(trailing_silence_ms))
    _speech_ratio = max(0.0, min(1.0, float(speech_ratio)))
    _mean_rms = max(0.0, float(mean_rms))
    _max_rms = max(0.0, float(max_rms))

    if looks_like_short_cta_tail(value):
        return "cjk-tail-hallucination"

    weak_speech = _speech_ratio < 0.24
    weak_energy = _mean_rms < 0.008 and _max_rms < 0.030
    tail_dominant = _trailing_silence_ms >= max(650, int(effective_audio_ms * 0.55))
    tiny_uncertain_segment = effective_audio_ms <= 650 and _speech_ratio < 0.45
    weak_evidence = weak_speech or weak_energy or tail_dominant or tiny_uncertain_segment

    if normalized in _SHORT_TAIL_HALLUCINATION_NORMALIZED and weak_evidence:
        return "short-tail-hallucination"
    if any(pattern.match(compact) for pattern in _TAIL_HALLUCINATION_PATTERNS) and weak_evidence:
        return "short-tail-hallucination"
    if any(pattern.match(compact) for pattern in _CJK_TAIL_HALLUCINATION_PATTERNS) and (
        weak_evidence or effective_audio_ms <= 1200
    ):
        return "cjk-tail-hallucination"
    return ""


__all__ = [
    "_CJK_TAIL_HALLUCINATION_PATTERNS",
    "_SHORT_TAIL_HALLUCINATION_NORMALIZED",
    "_TAIL_HALLUCINATION_PATTERNS",
    "looks_like_repetitive_loop",
    "looks_like_short_cta_tail",
    "tail_hallucination_drop_reason",
    # --- transcript hallucination helpers ---
    "_HALLUCINATION_PATTERNS",
    "_NON_SPEECH_NORMALIZED_SUBSTRINGS",
    "_NON_SPEECH_TEXT_PATTERNS",
    "_SHORT_HALLUCINATION_NORMALIZED",
    "_format_asr_exception_message",
    "_looks_like_known_non_speech_text",
    "_looks_like_script_mismatch_junk",
    "_looks_like_silence_hallucination",
    "_transcript_drop_reason",
]


# ---------------------------------------------------------------------------
# Hallucination / non-speech helpers shared by ASR runtime and tests.
# ---------------------------------------------------------------------------

_HALLUCINATION_PATTERNS = (
    re.compile(r"^\s*thank(s| you)( everyone| all)?[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thank(s| you)?\s+for\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*thanks\s+for\s+your\s+watching[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*good night[.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*bye(-| )?bye[.! ]*$", re.IGNORECASE),
)

_SHORT_HALLUCINATION_NORMALIZED = {
    "bybwd6",
    "thankyou",
    "thanks",
    "thankyouall",
    "thankyoueveryone",
    "thankseveryone",
    "thanksall",
    "goodnight",
    "byebye",
    "謝謝大家",
    "谢谢大家",
    "感謝大家",
    "感谢大家",
    "晚安",
    "感謝您的收看",
    "感谢您的收看",
}

_NON_SPEECH_TEXT_PATTERNS = (
    re.compile(r"yoyo\s+television\s+series\s+exclusive", re.IGNORECASE),
    re.compile(r"amara\.org", re.IGNORECASE),
    re.compile(r"點贊.*訂閱.*轉發.*打賞", re.IGNORECASE),
    re.compile(r"ming\s+pao\s+canada\s+ming\s+pao\s+toronto", re.IGNORECASE),
    re.compile(r"謝謝觀看.*下次見", re.IGNORECASE),
)

_NON_SPEECH_NORMALIZED_SUBSTRINGS = (
    "yoyotelevisionseriesexclusive",
    "amaraorg",
    "點贊訂閱轉發打賞",
    "mingpaocanadamingpaotoronto",
    "謝謝觀看下次見",
)


def _format_asr_exception_message(exc: Exception) -> str:
    """Return a human-readable message for an ASR exception."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "dll load failed while importing _ssl" in lowered:
        return (
            f"{message} OpenSSL runtime is missing. "
            "Check _internal/libssl-3-x64.dll, _internal/libcrypto-3-x64.dll, "
            "_internal/_ssl.pyd, and dist/SyncTranslate-onedir/SyncTranslate.exe."
        )
    return message


def _looks_like_silence_hallucination(text: str, *, audio_ms: int, vad_rms: float) -> bool:
    """Return True when *text* looks like a silence-induced hallucination."""
    value = (text or "").strip()
    if not value:
        return False
    if audio_ms > 1800 or vad_rms >= 0.035:
        return False
    compact = re.sub(r"\s+", " ", value)
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    if normalized in _SHORT_HALLUCINATION_NORMALIZED:
        return True
    return any(pattern.match(compact) for pattern in _HALLUCINATION_PATTERNS)


def _looks_like_known_non_speech_text(text: str) -> bool:
    """Return True when *text* matches a known non-speech watermark or overlay."""
    value = (text or "").strip()
    if not value:
        return False
    compact = re.sub(r"\s+", " ", value)
    normalized = "".join(ch.lower() for ch in value if ch.isalnum())
    if any(token in normalized for token in _NON_SPEECH_NORMALIZED_SUBSTRINGS):
        return True
    return any(pattern.search(compact) for pattern in _NON_SPEECH_TEXT_PATTERNS)


def _looks_like_script_mismatch_junk(text: str, *, expected_language: str) -> bool:
    """Return True when *text* is a short run of characters whose script does not
    match *expected_language* (e.g. Cyrillic characters for a Chinese stream).
    """
    normalized_language = (expected_language or "").strip().lower()
    if not normalized_language:
        return False
    if "-" in normalized_language:
        normalized_language = normalized_language.split("-", 1)[0]

    compact = [ch for ch in (text or "").strip() if ch.isalpha()]
    if not compact or len(compact) > 6:
        return False
    if not contains_cyrillic_or_greek(compact):
        return False

    if normalized_language == "zh":
        return not any(is_cjk_char(ch) or is_latin_char(ch) for ch in compact)
    if normalized_language == "en":
        return not all(is_latin_char(ch) for ch in compact)
    if normalized_language == "ja":
        return not any(is_japanese_char(ch) or is_latin_char(ch) for ch in compact)
    if normalized_language == "ko":
        return not any(is_hangul_char(ch) or is_latin_char(ch) for ch in compact)
    return False


def _transcript_drop_reason(
    text: str,
    *,
    audio_ms: int,
    vad_rms: float,
    expected_language: str,
) -> str:
    """Return a non-empty reason string when *text* should be dropped.

    Checks non-speech overlays, silence hallucinations, and script mismatches
    in that priority order.
    """
    if _looks_like_known_non_speech_text(text):
        return "non-speech-overlay"
    if _looks_like_silence_hallucination(text, audio_ms=audio_ms, vad_rms=vad_rms):
        return "hallucinated"
    if _looks_like_script_mismatch_junk(text, expected_language=expected_language):
        return "script-mismatch"
    return ""
