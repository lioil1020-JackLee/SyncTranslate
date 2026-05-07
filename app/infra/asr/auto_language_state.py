from __future__ import annotations

from dataclasses import dataclass

# Number of consecutive same-family finals required to trigger a profile switch.
_SWITCH_STREAK = 3
# Minimum ms between two consecutive effective language switches (cooldown).
_SWITCH_COOLDOWN_MS = 25_000
# Minimum CJK character count in a final for it to qualify for streak counting.
_MIN_CHINESE_CHARS = 4
# Minimum Latin word count in a final for it to qualify for streak counting.
_MIN_LATIN_WORDS = 3
# CJK character ratio threshold: at or above this → treat utterance as Chinese.
_CJK_RATIO_CHINESE = 0.35
# Latin character ratio threshold when CJK < 0.05 → treat utterance as non-Chinese.
_LATIN_RATIO_NON_CHINESE = 0.60


@dataclass(slots=True)
class AutoLanguageState:
    """Per-source dynamic language tracking for ASR 'auto' mode.

    Active only when requested_language is '' or 'auto'.  The manager updates
    this state on every final event and may rebuild the source runtime if the
    effective_language changes.
    """

    requested_language: str = ""
    effective_language: str = ""
    last_detected_language: str = ""
    # 'auto' | 'chinese' | 'non_chinese'
    stable_family: str = "auto"
    chinese_streak: int = 0
    non_chinese_streak: int = 0
    mixed_count: int = 0
    # monotonic milliseconds of the last profile switch (0 = never switched)
    last_switch_ms: int = 0


# ---------------------------------------------------------------------------
# Character-ratio helpers
# ---------------------------------------------------------------------------

def _cjk_ratio(text: str) -> float:
    """Return fraction of non-whitespace chars that fall in CJK Unicode blocks."""
    total = 0
    cjk = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF  # Extension A
            or 0x20000 <= cp <= 0x2A6DF  # Extension B
            or 0x2A700 <= cp <= 0x2B73F  # Extension C
            or 0x2B740 <= cp <= 0x2B81F  # Extension D
            or 0xF900 <= cp <= 0xFAFF  # Compatibility Ideographs
            or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        ):
            cjk += 1
    return cjk / total if total > 0 else 0.0


def _latin_ratio(text: str) -> float:
    """Return fraction of non-whitespace chars that are ASCII alphabetic."""
    total = 0
    latin = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch.isascii() and ch.isalpha():
            latin += 1
    return latin / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _is_too_short(text: str, family: str) -> bool:
    """Return True when the utterance is too brief to drive a language switch."""
    stripped = text.strip()
    if not stripped:
        return True
    if family == "chinese":
        cjk_count = sum(
            1 for ch in stripped
            if 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF
        )
        return cjk_count < _MIN_CHINESE_CHARS
    word_count = len(stripped.split())
    return word_count < _MIN_LATIN_WORDS


def _classify_text_family(detected_language: str, text: str) -> str:
    """Classify an utterance as 'chinese', 'non_chinese', or 'mixed'.

    Uses CJK/Latin character ratios as the primary signal and falls back to
    the detected_language tag when the ratio is ambiguous.
    """
    if not text.strip():
        return "mixed"
    cjk = _cjk_ratio(text)
    latin = _latin_ratio(text)
    if cjk >= _CJK_RATIO_CHINESE:
        return "chinese"
    if latin >= _LATIN_RATIO_NON_CHINESE and cjk < 0.05:
        return "non_chinese"
    # Ratio is ambiguous – use detected_language as a tiebreaker.
    lang = str(detected_language or "").strip().lower()
    if lang in {"zh", "zh-cn", "zh-tw", "cmn", "yue"}:
        return "chinese"
    if lang and not lang.startswith("zh") and not lang.startswith("cmn"):
        return "non_chinese"
    return "mixed"


# ---------------------------------------------------------------------------
# Main state-update function
# ---------------------------------------------------------------------------

def observe_final_language(
    state: AutoLanguageState,
    detected_language: str,
    text: str,
    now_ms: int,
    *,
    cooldown_ms: int = _SWITCH_COOLDOWN_MS,
    streak_threshold: int = _SWITCH_STREAK,
) -> str | None:
    """Update *state* from one final ASR event.

    Returns the new effective_language string if a profile switch is triggered,
    or ``None`` when no switch is needed.

    Only active when *state.requested_language* is empty or 'auto'; for any
    other requested language the function is a no-op and returns ``None``.
    """
    requested = str(state.requested_language or "").strip().lower()
    if requested and requested != "auto":
        return None

    state.last_detected_language = str(detected_language or "").strip()
    family = _classify_text_family(detected_language, text)

    if _is_too_short(text, family):
        return None

    # Update streaks.
    if family == "chinese":
        state.chinese_streak += 1
        state.non_chinese_streak = 0
    elif family == "non_chinese":
        state.non_chinese_streak += 1
        state.chinese_streak = 0
    else:
        # Mixed utterances: gently deflate both streaks to resist flip-flop.
        state.mixed_count += 1
        state.chinese_streak = max(0, state.chinese_streak - 1)
        state.non_chinese_streak = max(0, state.non_chinese_streak - 1)

    # Enforce cooldown between consecutive switches.
    if state.last_switch_ms > 0 and (now_ms - state.last_switch_ms) < cooldown_ms:
        return None

    if family == "chinese" and state.chinese_streak >= streak_threshold:
        target = "zh-TW"
        if state.effective_language != target:
            state.effective_language = target
            state.stable_family = "chinese"
            state.last_switch_ms = now_ms
            state.chinese_streak = 0
            return target

    elif family == "non_chinese" and state.non_chinese_streak >= streak_threshold:
        target = ""
        if state.effective_language != target:
            state.effective_language = target
            state.stable_family = "non_chinese"
            state.last_switch_ms = now_ms
            state.non_chinese_streak = 0
            return target

    return None


__all__ = [
    "AutoLanguageState",
    "observe_final_language",
]
