"""Prompt building helpers for OpenAI-compatible translation providers."""
from __future__ import annotations

from app.infra.config.schema import TranslationProfileConfig


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


def _correction_response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "asr_correction_payload",
            "schema": {
                "type": "object",
                "properties": {
                    "correction": {"type": "string"},
                },
                "required": ["correction"],
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


def _parse_stop_tokens(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    rows = [row.strip() for row in text.replace("\r", "\n").split("\n")]
    tokens = [row for row in rows if row]
    if len(tokens) == 1 and "," in tokens[0]:
        tokens = [item.strip() for item in tokens[0].split(",") if item.strip()]
    return tokens[:8]
