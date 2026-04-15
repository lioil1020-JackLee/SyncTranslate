"""Tests for TranscriptPostProcessor."""
from __future__ import annotations

import pytest

from app.application.transcript_postprocessor import TranscriptPostProcessor, _normalize_fullwidth


class TestTranscriptPostProcessorDisabled:
    def test_passthrough_when_disabled(self):
        pp = TranscriptPostProcessor(enabled=False)
        assert pp.process_partial("local", "hello world") == "hello world"
        assert pp.process_final("local", "hello world") == "hello world"

    def test_passthrough_empty_string(self):
        pp = TranscriptPostProcessor(enabled=False)
        assert pp.process_partial("local", "") == ""
        assert pp.process_final("local", "") == ""


class TestTextNormalization:
    def setup_method(self):
        self.pp = TranscriptPostProcessor(enabled=True, partial_stabilization_enabled=False)

    def test_trim(self):
        assert self.pp.process_final("local", "  hello  ") == "hello"

    def test_multiple_spaces(self):
        assert self.pp.process_final("local", "hello   world") == "hello world"

    def test_fullwidth_digits(self):
        result = _normalize_fullwidth("１２３")
        assert result == "123"

    def test_fullwidth_letters(self):
        result = _normalize_fullwidth("ＡＢＣ")
        assert result == "ABC"

    def test_punctuation_space_injection(self):
        result = self.pp.process_final("local", "Hello,world. This is a test")
        # should add space after comma
        assert "Hello, world" in result

    def test_empty_string(self):
        assert self.pp.process_final("local", "") == ""


class TestPartialStabilization:
    def setup_method(self):
        self.pp = TranscriptPostProcessor(enabled=True, partial_stabilization_enabled=True)

    def test_stable_prefix_preserved_on_regression(self):
        # 第一次 partial 較長
        r1 = self.pp.process_partial("local", "Hello world this is a test", utterance_id="u1")
        # 第二次 partial 縮短（可能是抖動）
        r2 = self.pp.process_partial("local", "Hello world", utterance_id="u1")
        # 應保留較長版本
        assert r2 == r1

    def test_growing_partial_accepted(self):
        self.pp.process_partial("local", "Hello", utterance_id="u2")
        r2 = self.pp.process_partial("local", "Hello world", utterance_id="u2")
        assert r2 == "Hello world"

    def test_different_utterances_independent(self):
        self.pp.process_partial("local", "utterance one text here", utterance_id="u3")
        r = self.pp.process_partial("local", "completely different", utterance_id="u4")
        assert r == "completely different"

    def test_final_clears_state(self):
        self.pp.process_partial("local", "Hello world this is a test", utterance_id="u5")
        self.pp.process_final("local", "Hello world this is a final.", utterance_id="u5")
        # state should be cleared; new partial for same utterance should be accepted normally
        r = self.pp.process_partial("local", "New partial", utterance_id="u5")
        assert r == "New partial"

    def test_reset_source(self):
        self.pp.process_partial("local", "text one two three", utterance_id="u6")
        self.pp.reset_source("local")
        r = self.pp.process_partial("local", "short", utterance_id="u6")
        assert r == "short"


class TestGlossaryIntegration:
    def test_glossary_applied_on_final(self):
        from app.domain.glossary import GlossaryEntry, GlossaryStore
        store = GlossaryStore([GlossaryEntry(pattern="chat gpt", replace="ChatGPT")])
        pp = TranscriptPostProcessor(
            enabled=True,
            partial_stabilization_enabled=False,
            glossary=store,
            glossary_apply_on_partial=False,
            glossary_apply_on_final=True,
        )
        result = pp.process_final("local", "I use chat gpt daily")
        assert "ChatGPT" in result
        assert "chat gpt" not in result.lower() or "ChatGPT" in result

    def test_glossary_not_applied_on_partial_when_disabled(self):
        from app.domain.glossary import GlossaryEntry, GlossaryStore
        store = GlossaryStore([GlossaryEntry(pattern="chat gpt", replace="ChatGPT")])
        pp = TranscriptPostProcessor(
            enabled=True,
            partial_stabilization_enabled=False,
            glossary=store,
            glossary_apply_on_partial=False,
            glossary_apply_on_final=True,
        )
        result = pp.process_partial("local", "I use chat gpt")
        assert "chat gpt" in result.lower()

    def test_no_glossary_ok(self):
        pp = TranscriptPostProcessor(enabled=True, glossary=None)
        result = pp.process_final("local", "Hello world")
        assert result == "Hello world"
