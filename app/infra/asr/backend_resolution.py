from __future__ import annotations

from dataclasses import dataclass


_CHINESE_LANGUAGE_CODES = {
    "zh",
    "zh-cn",
    "zh_cn",
    "zh-tw",
    "zh_tw",
    "cmn",
    "cmn-hans",
    "cmn-hant",
    "yue",
}


@dataclass(slots=True)
class BackendResolution:
    requested_language: str
    normalized_language: str
    language_family: str
    backend_name: str
    reason: str
    disabled: bool = False


def resolve_backend_for_language(language: str) -> BackendResolution:
    requested = str(language or "").strip()
    normalized = requested.lower().replace("_", "-")
    if not normalized:
        return BackendResolution(
            requested_language=requested,
            normalized_language="auto",
            language_family="auto",
            backend_name="faster_whisper_v2",
            reason="language is empty; defaulting to auto routing via faster-whisper",
        )
    if normalized == "none":
        return BackendResolution(
            requested_language=requested,
            normalized_language="none",
            language_family="disabled",
            backend_name="disabled",
            reason="ASR is disabled for this channel",
            disabled=True,
        )
    if normalized == "auto":
        return BackendResolution(
            requested_language=requested,
            normalized_language="auto",
            language_family="auto",
            backend_name="faster_whisper_v2",
            reason="auto language keeps the compatibility path on faster-whisper",
        )
    if normalized in _CHINESE_LANGUAGE_CODES:
        return BackendResolution(
            requested_language=requested,
            normalized_language=normalized,
            language_family="chinese",
            backend_name="funasr_v2",
            reason=f"{requested or normalized} is treated as a Chinese-family ASR language",
        )
    return BackendResolution(
        requested_language=requested,
        normalized_language=normalized,
        language_family="non_chinese",
        backend_name="faster_whisper_v2",
        reason=f"{requested or normalized} is treated as a non-Chinese ASR language",
    )


__all__ = ["BackendResolution", "resolve_backend_for_language"]
