"""Response text parsing helpers for OpenAI-compatible translation providers."""
from __future__ import annotations

import html
import json
import re


from app.domain.unicode_utils import contains_cjk as _contains_cjk


def _looks_like_glossary(text: str) -> bool:
    lowered = text.lower()
    glossary_tokens = ("->", " or ", " / ", " implies ", " translated as ", "pinyin", "qítā", "gōngzuò", " here ", " depending on ")
    return any(token in lowered for token in glossary_tokens)


def _extract_rhs_candidate(text: str) -> str:
    value = text.strip()
    for sep in ("->", "=>", "：", ":"):
        if sep in value:
            left, right = value.split(sep, 1)
            left_ascii = sum(ch.isascii() and ch.isalpha() for ch in left)
            right_cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in right)
            if right_cjk and left_ascii:
                return right.strip()
    return value


def _zh_line_score(text: str) -> tuple[int, int, int]:
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    punctuation = sum(ch in ".,:;/-_()[]{}*" for ch in text)
    return (cjk, -ascii_letters, -punctuation)


def _strip_thinking_sections(text: str) -> str:
    value = text.strip()
    if "<think>" in value and "</think>" in value:
        while "<think>" in value and "</think>" in value:
            start = value.find("<think>")
            end = value.find("</think>", start)
            if end < 0:
                break
            value = (value[:start] + value[end + len("</think>"):]).strip()
    return value


def _extract_translation_from_json(text: str) -> str:
    return _extract_json_field(text, "translation")


def _extract_json_field(text: str, field_name: str) -> str:
    value = text.strip()
    if not value:
        return ""
    candidates = [value]
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, value[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            result = payload.get(field_name)
            if isinstance(result, str):
                return result.strip()
    return ""


def _sanitize_surface_text(text: str) -> str:
    value = html.unescape((text or "").strip())
    if not value:
        return ""
    value = re.sub(r"```[\s\S]*?```", " ", value)
    value = re.sub(r"</?[^>\n]{1,120}>", " ", value)
    value = value.replace("\\n", "\n")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(translation|output|result|target|source|翻譯|译文)\s*[:：]\s*", "", value, flags=re.IGNORECASE)
    return value.strip(" \t\n\r\"'`[]{}")


def _clean_correction_output(text: str) -> str:
    value = _sanitize_surface_text(text)
    if not value:
        return ""
    value = re.sub(r"^(?:json|correction)\s*[:：-]?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*(?:```(?:json)?|json)\s*$", "", value, flags=re.IGNORECASE)
    value = value.strip(" \t\n\r\"'`[]{}")
    if _looks_like_structured_reply(value):
        return ""
    return value


def _looks_like_structured_reply(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    suspicious_tokens = (
        "```",
        '{"',
        '"}',
        '"correction"',
        '"translation"',
        "<think",
        "</think",
        "assistant:",
        "user:",
    )
    if any(token in lowered for token in suspicious_tokens):
        return True
    if re.search(r"(?:^|\s)json(?:\s|$)", lowered):
        return True
    if "{" in value or "}" in value:
        return True
    return False


def _looks_like_markup_fragment(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if re.search(r"</?[^>\n]{1,120}>", value):
        return True
    angle_pairs = value.count("<") + value.count(">")
    slash_pairs = value.count("/")
    return angle_pairs >= 2 and slash_pairs >= 1


def _looks_like_overexpanded_translation(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    banned_phrases = (
        "我會成為你的朋友",
        "我也愛你",
        "我能感受到你的存在",
        "非常感謝你",
    )
    return any(token in value for token in banned_phrases)


def _language_label(code: str) -> str:
    normalized = (code or "").strip().lower()
    mapping = {
        "zh": "Traditional Chinese",
        "zh-tw": "Traditional Chinese",
        "en": "English",
        "en-us": "English",
        "ja": "Japanese",
        "ja-jp": "Japanese",
        "ko": "Korean",
        "ko-kr": "Korean",
        "th": "Thai",
        "th-th": "Thai",
    }
    return mapping.get(normalized, code or "target language")
