"""Glossary 詞彙表領域模型。

提供專有名詞修正能力，支援 exact / substring / case-insensitive 匹配。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class GlossaryEntry:
    """單一詞彙替換規則。"""
    pattern: str
    replace: str
    case_sensitive: bool = False
    mode: str = "substring"  # "exact" | "substring"


class GlossaryStore:
    """持有並套用一組 GlossaryEntry。

    Parameters
    ----------
    entries:
        詞彙替換規則列表。
    """

    def __init__(self, entries: list[GlossaryEntry] | None = None) -> None:
        self._entries: list[GlossaryEntry] = []
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        if entries:
            self.load(entries)

    def load(self, entries: list[GlossaryEntry]) -> None:
        """載入（覆蓋）整批詞彙規則，並預編譯 regex。"""
        self._entries = list(entries)
        self._patterns = []
        for entry in self._entries:
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            if entry.mode == "exact":
                pat = re.compile(r"(?<!\w)" + re.escape(entry.pattern) + r"(?!\w)", flags)
            else:
                pat = re.compile(re.escape(entry.pattern), flags)
            self._patterns.append((pat, entry.replace))

    def apply(self, text: str, *, conservative: bool = False) -> str:
        """套用所有詞彙替換規則。

        Parameters
        ----------
        text:
            原始文字。
        conservative:
            True 時僅套用 exact 模式規則（用於 partial）；
            False 時套用全部規則（用於 final）。
        """
        if not text or not self._patterns:
            return text
        result = text
        for i, (pat, replacement) in enumerate(self._patterns):
            entry = self._entries[i]
            if conservative and entry.mode != "exact":
                continue
            result = pat.sub(replacement, result)
        return result

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["GlossaryEntry", "GlossaryStore"]
