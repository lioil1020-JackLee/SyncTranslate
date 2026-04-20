"""TranscriptPostProcessor — ASR 文字後處理層。

負責 partial stabilization、text normalization 與 glossary 修正，
可獨立啟用/停用，不影響 utterance_id / revision 流程。
"""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

try:
    from opencc import OpenCC  # type: ignore
except Exception:
    OpenCC = None

if TYPE_CHECKING:
    from app.domain.glossary import GlossaryStore


if OpenCC is not None:
    _S2T_CONVERTER = OpenCC("s2twp")
else:
    _S2T_CONVERTER = None


class TranscriptPostProcessor:
    """對 ASR partial / final 文字做後處理。

    Parameters
    ----------
    enabled:
        總開關，False 時 process_partial / process_final 直接回傳原文。
    partial_stabilization_enabled:
        啟用 partial stable prefix 比對，降低畫面抖動。
    glossary:
        可選的 GlossaryStore；None 表示不做詞彙修正。
    glossary_apply_on_partial:
        是否對 partial 套用 glossary（預設保守，關閉）。
    glossary_apply_on_final:
        是否對 final 套用 glossary（預設啟用）。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        partial_stabilization_enabled: bool = True,
        glossary: "GlossaryStore | None" = None,
        glossary_apply_on_partial: bool = False,
        glossary_apply_on_final: bool = True,
    ) -> None:
        self._enabled = enabled
        self._partial_stabilization_enabled = partial_stabilization_enabled
        self._glossary = glossary
        self._glossary_apply_on_partial = glossary_apply_on_partial
        self._glossary_apply_on_final = glossary_apply_on_final
        # {channel_key: last_normalized_partial_text}（用於 _stabilize_partial 比對）
        self._last_partial: dict[str, str] = {}
        # {channel_key: last_displayed_partial_text}（stabilize 後實際顯示的文字，用於 final 前綴回收）
        self._last_displayed_partial: dict[str, str] = {}
        # {source: last_final_text}（用於抑制跨 utterance 的短尾句重覆）
        self._last_final: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_partial(
        self,
        source: str,
        text: str,
        *,
        language: str = "",
        utterance_id: str = "",
    ) -> str:
        """後處理 partial transcript。回傳穩定化、正規化後的文字。"""
        if not self._enabled:
            return text

        normalized = self._normalize(text, language=language)

        if self._partial_stabilization_enabled:
            key = f"{source}:{utterance_id}"
            stabilized = self._stabilize_partial(key, normalized)
        else:
            key = f"{source}:{utterance_id}"
            stabilized = normalized

        # 記錄實際顯示的 partial，供 process_final 回收遺失前綴使用
        self._last_displayed_partial[key] = stabilized

        if self._glossary and self._glossary_apply_on_partial:
            stabilized = self._glossary.apply(stabilized, conservative=True)

        return stabilized

    def process_final(
        self,
        source: str,
        text: str,
        *,
        language: str = "",
        utterance_id: str = "",
    ) -> str:
        """後處理 final transcript。回傳正規化、glossary 修正後的文字。"""
        if not self._enabled:
            return text

        # 清除此 utterance 的 partial 狀態，並取出最後顯示的 partial 以供前綴回收
        key = f"{source}:{utterance_id}"
        self._last_partial.pop(key, None)
        last_displayed = self._last_displayed_partial.pop(key, "")

        normalized = self._normalize(text, language=language)

        # 若 final 比最後顯示的 partial 短很多，嘗試回收遺失的前綴
        # （ASR final 因 audio history 截斷，可能只有末尾幾秒的文字）
        if last_displayed and self._partial_stabilization_enabled:
            normalized = self._recover_final_prefix(last_displayed, normalized)

        # 壓縮連續重複片段，避免長段落出現 runaway hallucination 迴圈。
        normalized = _suppress_runaway_repetition(normalized, language=language)

        # 若新的 final 只是前一個 final 的短尾句重覆，直接抑制。
        normalized = self._suppress_short_cross_final_repeat(source, normalized)

        if self._glossary and self._glossary_apply_on_final:
            normalized = self._glossary.apply(normalized, conservative=False)

        if normalized:
            self._last_final[source] = normalized

        return normalized

    def reset_utterance(self, source: str, utterance_id: str) -> None:
        """主動清除指定 utterance 的暫存狀態。"""
        key = f"{source}:{utterance_id}"
        self._last_partial.pop(key, None)
        self._last_displayed_partial.pop(key, None)

    def reset_source(self, source: str) -> None:
        """清除特定 source 所有 utterance 的暫存狀態。"""
        prefix = f"{source}:"
        to_del = [k for k in self._last_partial if k.startswith(prefix)]
        for k in to_del:
            del self._last_partial[k]
        to_del = [k for k in self._last_displayed_partial if k.startswith(prefix)]
        for k in to_del:
            del self._last_displayed_partial[k]
        self._last_final.pop(source, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recover_final_prefix(self, last_displayed: str, final_text: str) -> str:
        """若 final 比最後顯示的 partial 短很多，嘗試從 partial 回收遺失的前綴。

        ASR final 因 audio history 截斷（final_history_seconds），可能只包含
        末尾幾秒的文字，而稍早已穩定顯示的前綴部分就會消失。
        此方法嘗試比對 final_text 是否為 last_displayed 的尾綴，若是則補回前綴。
        """
        if not last_displayed or not final_text:
            return final_text

        # final 已夠長（>= 80% of last_displayed），不需回收
        if len(final_text) >= len(last_displayed) * 0.8:
            return final_text

        # 取 final_text 前 30 個字元作為探針，在 last_displayed 中搜尋出現位置
        probe_len = min(30, len(final_text))
        probe = final_text[:probe_len].lower()
        haystack = last_displayed.lower()

        idx = haystack.find(probe)
        if idx <= 0:
            # 找不到或在最前面，表示 final 是全新改寫，不回收
            return final_text

        # 確認 last_displayed[idx:] 和 final_text 開頭確實對得上（至少 60%）
        overlap_len = min(len(last_displayed) - idx, len(final_text))
        match_chars = sum(
            1 for a, b in zip(last_displayed[idx:idx + overlap_len], final_text[:overlap_len])
            if a.lower() == b.lower()
        )
        if match_chars < overlap_len * 0.6:
            return final_text

        # 補回前綴
        prefix = last_displayed[:idx].rstrip()
        if not prefix:
            return final_text
        # 決定分隔符：CJK 不加空格，西文加空格
        needs_space = (
            prefix
            and final_text
            and prefix[-1].isascii()
            and prefix[-1] not in ",.!?;:"
            and final_text[0].isascii()
            and not final_text[0].isspace()
        )
        sep = " " if needs_space else ""
        return prefix + sep + final_text

    def _stabilize_partial(self, key: str, text: str) -> str:
        """若新 partial 與舊 partial 共享長前綴，保留穩定前綴避免抖動。"""
        prev = self._last_partial.get(key, "")
        self._last_partial[key] = text

        if not prev or not text:
            return text

        # 找最長共同前綴
        min_len = min(len(prev), len(text))
        prefix_len = 0
        for i in range(min_len):
            if prev[i] == text[i]:
                prefix_len = i + 1
            else:
                break

        # 若共同前綴超過較短字串的 60%（新文字是舊文字的前綴候選），
        # 且新文字比舊文字短很多（可能是抖動），傾向保留較長的版本
        shorter_len = min(len(prev), len(text))
        if prefix_len >= shorter_len * 0.6 and len(text) < len(prev) * 0.8:
            return prev

        return text

    def _suppress_short_cross_final_repeat(self, source: str, text: str) -> str:
        prev = self._last_final.get(source, "").strip()
        current = str(text or "").strip()
        if not prev or not current:
            return current

        current_compact = re.sub(r"\s+", "", current)
        prev_compact = re.sub(r"\s+", "", prev)
        if not current_compact or len(current_compact) > 12:
            return current

        tail_window = prev_compact[-max(len(current_compact) * 2, len(current_compact) + 6):]
        if prev_compact.endswith(current_compact) or current_compact in tail_window:
            return ""
        return current

    @staticmethod
    def _normalize(text: str, *, language: str = "") -> str:
        """基礎文字正規化。"""
        if not text:
            return text

        # 1. trim
        result = text.strip()

        # 2. 重複空白整理
        result = re.sub(r"  +", " ", result)

        # 3. 全形半形基本整理（數字 / 英文字母）
        result = _normalize_fullwidth(result)

        # 3.5 中文簡繁整理：中文 ASR 後端常輸出簡體，UI 以繁體中文為主。
        result = _normalize_chinese_script(result, language=language)

        # 4. 中英文標點整理（英文句末加空格）
        result = _normalize_punctuation(result, language=language)

        return result


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_FULLWIDTH_TABLE = str.maketrans(
    "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
    "!\"#$%&'()*+,-./"
    "0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
)


def _normalize_fullwidth(text: str) -> str:
    """將全形 ASCII 字元轉為半形。"""
    return text.translate(_FULLWIDTH_TABLE)


def _normalize_chinese_script(text: str, *, language: str = "") -> str:
    """將中文輸出統一為繁體（若 OpenCC 可用）。"""
    normalized_language = str(language or "").strip().lower().replace("_", "-")
    if _S2T_CONVERTER is None:
        return text
    if normalized_language in {"zh", "zh-tw", "zh-cn", "cmn", "cmn-hans", "cmn-hant", "yue"}:
        try:
            return _S2T_CONVERTER.convert(text)
        except Exception:
            return text
    return text


_PUNCT_SPACE_RE = re.compile(r"([.!?,;:])([A-Za-z\u4e00-\u9fff])")


def _normalize_punctuation(text: str, *, language: str = "") -> str:
    """在英文標點後缺少空格時補上空格（中文標點不變動）。"""
    return _PUNCT_SPACE_RE.sub(r"\1 \2", text)


def _suppress_runaway_repetition(text: str, *, language: str = "") -> str:
    """壓縮連續重複 n-gram，抑制 final 的爆詞迴圈。"""
    if not text or len(text) < 12:
        return text

    normalized_language = str(language or "").strip().lower().replace("_", "-")
    is_cjk = normalized_language in {"zh", "zh-tw", "zh-cn", "cmn", "cmn-hans", "cmn-hant", "yue"}

    if not is_cjk and " " in text:
        collapsed = _collapse_repeated_word_ngrams(text, max_ngram_words=8, min_repeats=3)
        collapsed = _collapse_repeated_char_ngrams(collapsed, min_ngram_chars=4, max_ngram_chars=24, min_repeats=3)
        return collapsed

    # CJK: 先處理短迴圈（n=1~3，需 6 次以上），再處理長片語（n=4~24，3 次以上）
    collapsed = _collapse_repeated_char_ngrams(text, min_ngram_chars=1, max_ngram_chars=3, min_repeats=6)
    collapsed = _collapse_repeated_char_ngrams(collapsed, min_ngram_chars=4, max_ngram_chars=24, min_repeats=3)
    return collapsed


def _collapse_repeated_char_ngrams(
    text: str,
    *,
    min_ngram_chars: int,
    max_ngram_chars: int,
    min_repeats: int,
) -> str:
    if len(text) < min_ngram_chars * min_repeats:
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        max_len = min(max_ngram_chars, (n - i) // min_repeats)
        for gram_len in range(max_len, min_ngram_chars - 1, -1):
            gram = text[i:i + gram_len]
            repeats = 1
            j = i + gram_len
            while j + gram_len <= n and text[j:j + gram_len] == gram:
                repeats += 1
                j += gram_len
            if repeats >= min_repeats:
                out.append(gram)
                i = j
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def _collapse_repeated_word_ngrams(
    text: str,
    *,
    max_ngram_words: int,
    min_repeats: int,
) -> str:
    tokens = text.split()
    if len(tokens) < max(4, min_repeats):
        return text

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        max_words = min(max_ngram_words, (n - i) // min_repeats)
        for gram_words in range(max_words, 0, -1):
            gram = tokens[i:i + gram_words]
            repeats = 1
            j = i + gram_words
            while j + gram_words <= n and tokens[j:j + gram_words] == gram:
                repeats += 1
                j += gram_words
            if repeats >= min_repeats:
                out.extend(gram)
                i = j
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


__all__ = ["TranscriptPostProcessor"]
