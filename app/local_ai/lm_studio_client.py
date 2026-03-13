from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from app.schemas import TranslationProfileConfig


@dataclass(slots=True)
class LmStudioClient:
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen/qwen3.5-9b"
    temperature: float = 0.2
    top_p: float = 0.9
    request_timeout_sec: float = 20.0

    def health_check(self) -> tuple[bool, str]:
        try:
            self._chat_completion(messages=[{"role": "user", "content": "ping"}], max_tokens=2)
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def list_models(self) -> list[str]:
        payload = self._request_json("/v1/models", method="GET")
        data = payload.get("data", [])
        return [str(item.get("id", "")) for item in data if isinstance(item, dict)]

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        context: list[str] | None = None,
        profile: TranslationProfileConfig | None = None,
    ) -> str:
        if not text.strip():
            return ""
        context_text = "\n".join((context or [])[-6:])
        source_label = _language_label(source_lang)
        target_label = _language_label(target_lang)
        style_hint = _profile_hint(profile)
        system_prompt = (
            "You are a real-time interpretation engine.\n"
            f"Translate only from {source_label} to {target_label}.\n"
            f"Return JSON only in this exact format: {{\"translation\":\"...\"}} where the value is only {target_label}.\n"
            "Do not output any other keys.\n"
            "Do not explain, analyze, answer, summarize, add notes, add bullet points, use markdown, or show thinking process.\n"
            "Never output Thinking Process, Analysis, Notes, romanization, pinyin, or source-language quotes.\n"
            "If the input is a fragment, translate it as a fragment.\n"
            "Keep named entities and numbers accurate.\n"
            f"Style policy: {style_hint}"
        )
        user_prompt = (
            f"Source language: {source_label}\n"
            f"Target language: {target_label}\n"
            "Context below is reference only. Do not translate the context by itself.\n"
            f"Context:\n{context_text}\n\n"
            "Translate this text only:\n"
            f"{text}"
        )
        response = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            response_format=_translation_response_format(),
        )
        cleaned = self._extract_translation_text(response, target_lang=target_lang)
        if self._looks_like_reasoning(cleaned, target_lang=target_lang):
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": response,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Return only valid JSON in the format {{\"translation\":\"...\"}}. "
                            f"The translation value must be only {target_label}, with no explanation."
                        ),
                    },
                ],
                max_tokens=128,
                response_format=_translation_response_format(),
            )
            cleaned = self._extract_translation_text(response, target_lang=target_lang)
        return cleaned

    def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        response_format: dict[str, object] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            data = self._request_json("/v1/chat/completions", payload=payload)
        except ValueError as exc:
            if response_format and "response_format.type" in str(exc):
                payload["response_format"] = {"type": "text"}
                data = self._request_json("/v1/chat/completions", payload=payload)
            else:
                raise
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

    @staticmethod
    def _extract_translation_text(text: str, *, target_lang: str) -> str:
        stripped = _strip_thinking_sections(text)
        json_translation = _extract_translation_from_json(stripped)
        if json_translation:
            return LmStudioClient._clean_translation_output(json_translation, target_lang=target_lang)
        return LmStudioClient._clean_translation_output(stripped, target_lang=target_lang)

    @staticmethod
    def _clean_translation_output(text: str, *, target_lang: str) -> str:
        lines = [line.strip(" -*\t") for line in text.replace("\r", "\n").split("\n")]
        filtered: list[str] = []
        banned_markers = (
            "thinking process",
            "analysis",
            "interpretation",
            "context",
            "translation:",
            "rewrite:",
            "analyze the request",
            "common phrasing",
            "implies",
            "depending on",
        )
        for line in lines:
            if not line:
                continue
            line = line.strip("[]\"'")
            lowered = line.lower()
            if any(marker in lowered for marker in banned_markers):
                continue
            line = _extract_rhs_candidate(line)
            if target_lang.lower().startswith("zh") and _looks_like_glossary(line):
                continue
            filtered.append(line)
        if not filtered:
            return ""
        if target_lang.lower().startswith("zh"):
            prioritized = sorted(
                [line for line in filtered if _contains_cjk(line) and not _looks_like_glossary(line)],
                key=_zh_line_score,
                reverse=True,
            )
            if prioritized:
                return prioritized[0].strip(" \"'")
        return filtered[0].strip(" \"'")

    @staticmethod
    def _looks_like_reasoning(text: str, *, target_lang: str) -> bool:
        lowered = text.lower()
        reasoning_tokens = (
            "interpretation",
            "context",
            "however",
            "actually",
            "translation",
            "let's",
            "means",
        )
        if any(token in lowered for token in reasoning_tokens):
            return True
        if "\n" in text:
            return True
        if _looks_like_glossary(text):
            return True
        if target_lang.lower().startswith("zh") and not _contains_cjk(text):
            return True
        return False

    def _request_json(self, path: str, payload: dict[str, object] | None = None, method: str = "POST") -> dict:
        url = self.base_url.rstrip("/") + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"local llm connection failed: {reason}") from exc
        if not content.strip():
            return {}
        return json.loads(content)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


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
            translation = payload.get("translation")
            if isinstance(translation, str):
                return translation.strip()
    return ""


def _language_label(code: str) -> str:
    normalized = (code or "").strip().lower()
    mapping = {
        "zh": "Traditional Chinese",
        "zh-tw": "Traditional Chinese",
        "en": "English",
        "en-us": "English",
    }
    return mapping.get(normalized, code or "target language")


def _translation_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "translation_payload",
            "schema": {
                "type": "object",
                "properties": {
                    "translation": {"type": "string"},
                },
                "required": ["translation"],
                "additionalProperties": False,
            },
        },
    }


def _profile_hint(profile: TranslationProfileConfig | None) -> str:
    if profile is None:
        return "literal and concise"
    rules: list[str] = []
    if profile.prompt_style:
        rules.append(f"style={profile.prompt_style}")
    rules.append("preserve terms" if profile.preserve_terms else "allow term simplification")
    rules.append("natural tone" if profile.naturalize_tone else "neutral tone")
    rules.append(
        "allow light subject completion" if profile.allow_subject_completion else "avoid adding implied subjects"
    )
    return ", ".join(rules)