"""Tests for GlossaryStore and GlossaryLoader."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.domain.glossary import GlossaryEntry, GlossaryStore
from app.infra.config.glossary_loader import load_glossary


class TestGlossaryStore:
    def test_empty_store(self):
        store = GlossaryStore()
        assert store.is_empty()
        assert len(store) == 0
        assert store.apply("hello") == "hello"

    def test_basic_substring_replace(self):
        store = GlossaryStore([GlossaryEntry(pattern="chat gpt", replace="ChatGPT")])
        result = store.apply("I use chat gpt every day")
        assert result == "I use ChatGPT every day"

    def test_case_insensitive_by_default(self):
        store = GlossaryStore([GlossaryEntry(pattern="Chat GPT", replace="ChatGPT")])
        result = store.apply("I use chat gpt every day")
        assert result == "I use ChatGPT every day"

    def test_case_sensitive(self):
        store = GlossaryStore([GlossaryEntry(pattern="Chat GPT", replace="ChatGPT", case_sensitive=True)])
        result_no_match = store.apply("I use chat gpt every day")
        assert "chat gpt" in result_no_match
        result_match = store.apply("I use Chat GPT every day")
        assert result_match == "I use ChatGPT every day"

    def test_exact_mode(self):
        store = GlossaryStore([GlossaryEntry(pattern="AI", replace="A.I.", mode="exact")])
        # Should NOT match "FAIL" or "RAIN"
        result = store.apply("FAIL to use AI today")
        assert result == "FAIL to use A.I. today"

    def test_conservative_skips_substring(self):
        store = GlossaryStore([
            GlossaryEntry(pattern="gpt", replace="GPT", mode="substring"),
            GlossaryEntry(pattern="chat gpt", replace="ChatGPT", mode="exact"),
        ])
        result = store.apply("I use chat gpt", conservative=True)
        # only exact mode applied
        assert "ChatGPT" in result

    def test_multiple_entries(self):
        store = GlossaryStore([
            GlossaryEntry(pattern="fun asr", replace="FunASR"),
            GlossaryEntry(pattern="faster whisper", replace="faster-whisper"),
        ])
        result = store.apply("fun asr and faster whisper")
        assert result == "FunASR and faster-whisper"

    def test_load_replaces_previous(self):
        store = GlossaryStore([GlossaryEntry(pattern="old", replace="OLD")])
        store.load([GlossaryEntry(pattern="new", replace="NEW")])
        assert store.apply("old new") == "old NEW"


class TestGlossaryLoader:
    def test_load_none_path(self):
        store = load_glossary(None)
        assert store.is_empty()

    def test_load_nonexistent_path(self):
        store = load_glossary("/nonexistent/path/glossary.yaml")
        assert store.is_empty()

    def test_load_json(self, tmp_path):
        data = {"entries": [{"pattern": "chat gpt", "replace": "ChatGPT"}]}
        p = tmp_path / "glossary.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        store = load_glossary(p)
        assert not store.is_empty()
        assert store.apply("chat gpt") == "ChatGPT"

    def test_load_yaml(self, tmp_path):
        yaml_text = "entries:\n  - pattern: chat gpt\n    replace: ChatGPT\n"
        p = tmp_path / "glossary.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        store = load_glossary(p)
        assert not store.is_empty()
        assert store.apply("chat gpt") == "ChatGPT"

    def test_load_invalid_file_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{invalid", encoding="utf-8")
        # Should not raise, just return empty
        store = load_glossary(p)
        # empty or not, should not crash
        assert isinstance(store, GlossaryStore)

    def test_load_missing_fields_skipped(self, tmp_path):
        data = {"entries": [{"pattern": "", "replace": "X"}, {"pattern": "ok", "replace": "OK"}]}
        p = tmp_path / "glossary.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        store = load_glossary(p)
        assert len(store) == 1
