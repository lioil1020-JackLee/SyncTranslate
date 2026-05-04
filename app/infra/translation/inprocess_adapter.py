from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from threading import Lock, RLock
from typing import Any

from app.infra.config.schema import DEFAULT_FIXED_LLM_MODEL, TranslationProfileConfig
from app.infra.translation._stream_parser import (
    _contains_cjk,
    _looks_like_glossary,
    _extract_rhs_candidate,
    _zh_line_score,
    _strip_thinking_sections,
    _extract_translation_from_json,
    _extract_json_field,
    _sanitize_surface_text,
    _clean_correction_output,
    _looks_like_structured_reply,
    _looks_like_markup_fragment,
    _looks_like_overexpanded_translation,
    _language_label,
)
from app.infra.translation._prompt_builder import (
    _translation_response_format,
    _correction_response_format,
    _profile_hint,
    _parse_stop_tokens,
)


@dataclass(slots=True)
class _RuntimeHandle:
    model_path: str
    llm: Any
    lock: RLock = field(default_factory=RLock)


_RUNTIME_CACHE: dict[tuple[str, int, int, int, int], _RuntimeHandle] = {}
_RUNTIME_CACHE_LOCK = Lock()


def _build_runtime(*, model_path: str, ctx_size: int, gpu_layers: int, threads: int, batch_size: int) -> _RuntimeHandle:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: .\\tools\\runtime_setup\\prepare_external_runtimes.ps1 to install all dependencies."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to initialize llama-cpp-python: {exc}"
        ) from exc

    resolved = Path(model_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"LLM model file not found: {resolved}")

    kwargs: dict[str, object] = {
        "model_path": str(resolved),
        "n_ctx": max(512, int(ctx_size)),
        "n_gpu_layers": max(0, int(gpu_layers)),
        "n_threads": max(1, int(threads)),
        "verbose": False,
    }
    if int(batch_size) > 0:
        kwargs["n_batch"] = int(batch_size)
    llm = Llama(**kwargs)
    return _RuntimeHandle(model_path=str(resolved), llm=llm)


def _get_runtime(*, model_path: str, ctx_size: int, gpu_layers: int, threads: int, batch_size: int) -> _RuntimeHandle:
    key = (
        str(Path(model_path).resolve()),
        max(512, int(ctx_size)),
        max(0, int(gpu_layers)),
        max(1, int(threads)),
        max(1, int(batch_size)),
    )
    with _RUNTIME_CACHE_LOCK:
        handle = _RUNTIME_CACHE.get(key)
        if handle is not None:
            return handle
        created = _build_runtime(
            model_path=model_path,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            threads=threads,
            batch_size=batch_size,
        )
        _RUNTIME_CACHE[key] = created
        return created


@dataclass(slots=True)
class InProcessLlamaClient:
    model_path: str
    model: str = DEFAULT_FIXED_LLM_MODEL
    ctx_size: int = 4096
    gpu_layers: int = 35
    threads: int = 8
    batch_size: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 128
    repeat_penalty: float = 1.05
    stop_tokens: str = "</target>\nTranslation:"
    request_timeout_sec: float = 20.0
    _last_raw_response: str = ""
    _last_cleaned_response: str = ""
    _last_error: str = ""
    _runtime: _RuntimeHandle | None = None

    def __post_init__(self) -> None:
        pass  # lazy: runtime loaded on first use

    def _ensure_runtime(self) -> _RuntimeHandle:
        if self._runtime is None:
            self._runtime = _get_runtime(
                model_path=self.model_path,
                ctx_size=self.ctx_size,
                gpu_layers=self.gpu_layers,
                threads=self.threads,
                batch_size=self.batch_size,
            )
        return self._runtime

    def health_check(self) -> tuple[bool, str]:
        try:
            self._chat_completion(messages=[{"role": "user", "content": "ping"}], max_tokens=2)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            return False, str(exc)

    def list_models(self) -> list[str]:
        return [self.model]

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
        self._last_raw_response = ""
        self._last_cleaned_response = ""
        self._last_error = ""
        context_text = "\n".join((context or [])[-6:])
        source_label = _language_label(source_lang)
        target_label = _language_label(target_lang)
        style_hint = _profile_hint(profile)
        trad_chinese_note = (
            "IMPORTANT: The target language is Traditional Chinese. "
            "You MUST use Traditional Chinese characters only. "
            "Never use Simplified Chinese characters under any circumstances.\n"
        ) if "traditional chinese" in target_label.lower() else ""
        system_prompt = (
            "You are a real-time interpretation engine.\n"
            f"Translate only from {source_label} to {target_label}.\n"
            f"{trad_chinese_note}"
            f"Return JSON only in this exact format: {{\"translation\":\"...\"}} where the value is only {target_label}.\n"
            "Do not output any other keys.\n"
            "Do not explain, analyze, answer, summarize, add notes, add bullet points, use markdown, or show thinking process.\n"
            "Never output Thinking Process, Analysis, Notes, romanization, pinyin, or source-language quotes.\n"
            "If the input is a fragment, translate it as a fragment.\n"
            "Translate literally and conservatively.\n"
            "Do not continue the sentence, do not infer missing context, do not add emotions, and do not make the text longer than needed.\n"
            "If the source is short, keep the translation short.\n"
            "If the meaning is unclear, prefer a plain literal translation over a natural rewrite.\n"
            "Keep named entities and numbers accurate.\n"
            f"Style policy: {style_hint}"
        )
        if context_text.strip():
            user_prompt = (
                f"Source language: {source_label}\n"
                f"Target language: {target_label}\n"
                "Context below is reference only. Do not translate the context by itself.\n"
                f"Context:\n{context_text}\n\n"
                "Translate this text only:\n"
                f"{text}"
            )
        else:
            user_prompt = (
                f"Source language: {source_label}\n"
                f"Target language: {target_label}\n"
                "Translate this text only:\n"
                f"{text}"
            )
        response = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(16, int(self.max_output_tokens)),
            response_format=_translation_response_format(),
        )
        cleaned = self._extract_translation_text(response, target_lang=target_lang)
        self._last_raw_response = response
        self._last_cleaned_response = cleaned
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
                            "Return only valid JSON in the format {\"translation\":\"...\"}. "
                            f"The translation value must be only {target_label}, with no explanation."
                        ),
                    },
                ],
                max_tokens=max(16, min(128, int(self.max_output_tokens))),
                response_format=_translation_response_format(),
            )
            cleaned = self._extract_translation_text(response, target_lang=target_lang)
            self._last_raw_response = response
            self._last_cleaned_response = cleaned
        return cleaned

    def correct_asr_text(
        self,
        *,
        text: str,
        language: str,
        context: list[str] | None = None,
    ) -> str:
        if not text.strip():
            return ""
        self._last_raw_response = ""
        self._last_cleaned_response = ""
        self._last_error = ""
        language_label = _language_label(language or "")
        context_text = "\n".join((context or [])[-4:])
        system_prompt = (
            "You are an ASR post-editor.\n"
            f"Correct the transcript in {language_label}.\n"
            "Return JSON only in this exact format: {\"correction\":\"...\"}.\n"
            "Correct only obvious ASR mistakes.\n"
            "Prefer fixing homophone and near-sound recognition mistakes when the intended wording is clear.\n"
            "Do not translate.\n"
            "Do not summarize.\n"
            "Do not continue incomplete sentences.\n"
            "Do not add missing facts.\n"
            "Keep names, code, numbers, and product terms accurate.\n"
            "Keep sentence boundaries and punctuation stable unless they are clearly wrong.\n"
            "If the language is Traditional Chinese, use Traditional Chinese characters only.\n"
            "If uncertain, keep the original wording.\n"
            "Keep the output length close to the input."
        )
        if context_text.strip():
            user_prompt = (
                f"Reference context:\n{context_text}\n\n"
                "Correct this ASR text only:\n"
                f"{text}"
            )
        else:
            user_prompt = (
                "Correct this ASR text only:\n"
                f"{text}"
            )
        response = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(48, min(192, int(self.max_output_tokens))),
            response_format=_correction_response_format(),
        )
        cleaned = self._extract_correction_text(response)
        self._last_raw_response = response
        self._last_cleaned_response = cleaned
        return cleaned or text.strip()

    def debug_snapshot(self) -> dict[str, str]:
        return {
            "raw_response": self._trim_debug_text(self._last_raw_response),
            "cleaned_response": self._trim_debug_text(self._last_cleaned_response),
            "last_error": self._trim_debug_text(self._last_error),
        }

    def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        response_format: dict[str, object] | None = None,
    ) -> str:
        runtime = self._ensure_runtime()

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "repeat_penalty": self.repeat_penalty,
        }
        stop = _parse_stop_tokens(self.stop_tokens)
        if stop:
            kwargs["stop"] = stop
        if response_format:
            kwargs["response_format"] = response_format

        with runtime.lock:
            try:
                data = runtime.llm.create_chat_completion(**kwargs)
            except TypeError:
                kwargs.pop("response_format", None)
                data = runtime.llm.create_chat_completion(**kwargs)
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                raise ValueError(f"local in-process llm failed: {exc}") from exc

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()
        return str(first.get("text", "")).strip() if isinstance(first, dict) else ""

    @staticmethod
    def _extract_translation_text(text: str, *, target_lang: str) -> str:
        stripped = _strip_thinking_sections(text)
        json_translation = _extract_translation_from_json(stripped)
        if json_translation:
            return InProcessLlamaClient._clean_translation_output(json_translation, target_lang=target_lang)
        return InProcessLlamaClient._clean_translation_output(stripped, target_lang=target_lang)

    @staticmethod
    def _extract_correction_text(text: str) -> str:
        stripped = _strip_thinking_sections(text)
        json_correction = _extract_json_field(stripped, "correction")
        if json_correction:
            return _clean_correction_output(json_correction)
        if _looks_like_structured_reply(stripped):
            return ""
        return _clean_correction_output(stripped)

    @staticmethod
    def _clean_translation_output(text: str, *, target_lang: str) -> str:
        normalized = _sanitize_surface_text(text)
        lines = [line.strip(" -*\t") for line in normalized.replace("\r", "\n").split("\n")]
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
            "<translation",
            "</translation",
            "<target",
            "</target",
            "<p>",
            "</p>",
            "```",
        )
        for line in lines:
            if not line:
                continue
            line = _sanitize_surface_text(line)
            if not line:
                continue
            line = line.strip("[]\"'")
            lowered = line.lower()
            if any(marker in lowered for marker in banned_markers):
                continue
            line = _extract_rhs_candidate(line)
            if _looks_like_markup_fragment(line):
                continue
            if target_lang.lower().startswith("zh") and _looks_like_glossary(line):
                continue
            filtered.append(line)
        if not filtered:
            return ""
        if target_lang.lower().startswith("zh"):
            filtered = [line for line in filtered if not _looks_like_overexpanded_translation(line)]
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

    @staticmethod
    def _trim_debug_text(text: str, limit: int = 240) -> str:
        value = (text or "").strip().replace("\n", "\\n")
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."
