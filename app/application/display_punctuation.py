"""display_punctuation — lightweight CJK display-layer punctuation insertion.

Applies only at the UI rendering layer (not to the translation pipeline).
Rules are intentionally conservative to avoid false positives.
"""
from __future__ import annotations

import re

# Minimum CJK character count before appending sentence-ending punctuation.
_MIN_CJK_FOR_PERIOD = 8

# Question words that justify appending a full-width question mark.
_QUESTION_WORDS_ZH = re.compile(
    r"[嗎吧呢麼么什怎誰哪哪裡]$|^[什怎誰哪]"
)

# Characters that already carry terminal punctuation — skip if present.
_TERMINAL_PUNCT = frozenset("。？！…～…,.!?")

# Unicode CJK block ranges (covers CJK + CJK-A/B extensions + Compat ideographs).
_RE_CJK = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"
    r"\u3040-\u30FF\uAC00-\uD7AF]"
)


def _cjk_count(text: str) -> int:
    return len(_RE_CJK.findall(text))


def apply_display_punctuation(
    text: str,
    *,
    is_final: bool = True,
    enabled: bool = True,
) -> str:
    """Return *text* with display-only punctuation appended where appropriate.

    Only acts on CJK-heavy final segments.  Partials and non-CJK text pass
    through unchanged.
    """
    if not enabled or not is_final:
        return text

    stripped = text.strip()
    if not stripped:
        return text

    # Skip if text already ends with terminal punctuation.
    if stripped[-1] in _TERMINAL_PUNCT:
        return text

    cjk = _cjk_count(stripped)

    # Too short — likely a filler; don't add punctuation.
    if cjk < 4:
        return text

    # Question words → full-width question mark.
    if _QUESTION_WORDS_ZH.search(stripped):
        return text.rstrip() + "？"

    # Long enough CJK segment → full-width period.
    if cjk >= _MIN_CJK_FOR_PERIOD:
        return text.rstrip() + "。"

    return text


__all__ = ["apply_display_punctuation"]
