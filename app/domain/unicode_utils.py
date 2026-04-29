"""Unicode / script-family character helpers used across ASR and translation modules.

All functions operate on single characters or plain strings and have no side
effects.  Import from this module instead of duplicating range checks.
"""
from __future__ import annotations

import unicodedata


def is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK character.

    Covers CJK Extension A (U+3400–U+4DBF), CJK Unified Ideographs
    (U+4E00–U+9FFF), and CJK Compatibility Ideographs (U+F900–U+FAFF).
    """
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
    )


def is_hangul_char(ch: str) -> bool:
    """Return True if *ch* is a Hangul syllable (U+AC00–U+D7AF)."""
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7AF


def is_japanese_char(ch: str) -> bool:
    """Return True if *ch* is CJK or Kana (hiragana / katakana)."""
    cp = ord(ch)
    return is_cjk_char(ch) or 0x3040 <= cp <= 0x30FF


def is_latin_char(ch: str) -> bool:
    """Return True if *ch* is a Latin letter (basic ASCII + extended ranges)."""
    cp = ord(ch)
    return (
        0x0041 <= cp <= 0x005A  # A–Z
        or 0x0061 <= cp <= 0x007A  # a–z
        or 0x00C0 <= cp <= 0x024F  # Latin Extended-A / Extended-B
    )


def contains_cjk(text: str) -> bool:
    """Return True if *text* contains at least one CJK Unified Ideograph (U+4E00–U+9FFF)."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def contains_cyrillic_or_greek(chars: list[str]) -> bool:
    """Return True if any character in *chars* is Cyrillic or Greek."""
    for ch in chars:
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name or "GREEK" in name:
            return True
    return False


__all__ = [
    "contains_cjk",
    "contains_cyrillic_or_greek",
    "is_cjk_char",
    "is_hangul_char",
    "is_japanese_char",
    "is_latin_char",
]
