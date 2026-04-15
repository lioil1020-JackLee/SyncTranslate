from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


@dataclass(slots=True)
class HotwordEntry:
    alias: str
    canonical: str


class AsrLexicalBiaser:
    def __init__(self, raw_entries: str | list[str] | None, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._entries = self._parse_entries(raw_entries)

    def apply(self, text: str, *, language: str) -> str:
        value = str(text or "").strip()
        if not self._enabled or not value or not self._entries:
            return value
        normalized_language = str(language or "").strip().lower()
        result = value
        for entry in self._entries:
            if entry.alias in result:
                result = result.replace(entry.alias, entry.canonical)
                if entry.alias == entry.canonical:
                    continue
            fuzzy = self._replace_fuzzy(result, entry, language=normalized_language)
            if fuzzy != result:
                result = fuzzy
        return result

    @staticmethod
    def _parse_entries(raw_entries: str | list[str] | None) -> list[HotwordEntry]:
        if raw_entries is None:
            return []
        if isinstance(raw_entries, str):
            rows = [row.strip() for row in raw_entries.replace("\r", "\n").split("\n")]
            if len(rows) == 1 and "," in rows[0]:
                rows = [item.strip() for item in rows[0].split(",")]
        else:
            rows = [str(item or "").strip() for item in raw_entries]
        entries: list[HotwordEntry] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if not row:
                continue
            if "=>" in row:
                alias, canonical = row.split("=>", 1)
            elif "->" in row:
                alias, canonical = row.split("->", 1)
            else:
                alias = row
                canonical = row
            alias = alias.strip()
            canonical = canonical.strip()
            if not alias or not canonical:
                continue
            key = (alias, canonical)
            if key in seen:
                continue
            seen.add(key)
            entries.append(HotwordEntry(alias=alias, canonical=canonical))
        return entries

    def _replace_fuzzy(self, text: str, entry: HotwordEntry, *, language: str) -> str:
        alias = entry.alias
        canonical = entry.canonical
        if len(alias) < 2:
            return text
        if self._is_cjk_text(alias) or language.startswith(("zh", "ja", "ko", "yue", "cmn")):
            return self._replace_fuzzy_cjk(text, alias=alias, canonical=canonical)
        return self._replace_fuzzy_tokenized(text, alias=alias, canonical=canonical)

    @staticmethod
    def _replace_fuzzy_cjk(text: str, *, alias: str, canonical: str) -> str:
        best_start = -1
        best_end = -1
        best_score = 0.0
        min_len = max(1, len(alias) - 1)
        max_len = min(len(text), len(alias) + 1)
        for window_len in range(min_len, max_len + 1):
            for start in range(0, max(0, len(text) - window_len) + 1):
                candidate = text[start : start + window_len]
                score = AsrLexicalBiaser._score_cjk_candidate(candidate, alias)
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_end = start + window_len
        if best_score >= 0.78 and best_start >= 0:
            candidate = text[best_start:best_end]
            if candidate != alias:
                return text[:best_start] + canonical + text[best_end:]
        return text

    @staticmethod
    def _score_cjk_candidate(candidate: str, alias: str) -> float:
        if not candidate or not alias:
            return 0.0
        seq_ratio = SequenceMatcher(a=candidate, b=alias).ratio()
        positional = (
            sum(1 for left, right in zip(candidate, alias) if left == right)
            / float(max(len(candidate), len(alias), 1))
        )
        same_prefix = candidate[:2] == alias[:2]
        same_suffix = candidate[-1:] == alias[-1:]
        length_penalty = abs(len(candidate) - len(alias)) * 0.04
        single_substitution = AsrLexicalBiaser._is_single_cjk_substitution(candidate, alias)
        score = (seq_ratio * 0.65) + (positional * 0.35) - length_penalty
        if same_prefix:
            score += 0.06
        if same_suffix:
            score += 0.03
        if single_substitution:
            score += 0.08
        return score

    @staticmethod
    def _is_single_cjk_substitution(candidate: str, alias: str) -> bool:
        if len(candidate) != len(alias):
            return False
        mismatches = sum(1 for left, right in zip(candidate, alias) if left != right)
        return mismatches == 1

    @staticmethod
    def _replace_fuzzy_tokenized(text: str, *, alias: str, canonical: str) -> str:
        tokens = re.split(r"(\s+)", text)
        for idx, token in enumerate(tokens):
            stripped = token.strip()
            if not stripped or stripped == alias:
                continue
            ratio = SequenceMatcher(a=stripped.lower(), b=alias.lower()).ratio()
            if ratio >= 0.82:
                tokens[idx] = token.replace(stripped, canonical)
                break
        return "".join(tokens)

    @staticmethod
    def _is_cjk_text(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)


__all__ = ["AsrLexicalBiaser", "HotwordEntry"]
