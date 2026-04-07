from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from app.application.diagnostics_export import export_runtime_diagnostics, export_session_report
from app.infra.config.schema import AppConfig, AudioRouteConfig


class DiagnosticsExportTests(unittest.TestCase):
    def test_export_runtime_diagnostics_includes_runtime_modes_and_routes(self) -> None:
        cfg = AppConfig()
        cfg.direction.mode = "bidirectional"
        cfg.runtime.remote_translation_enabled = False
        cfg.runtime.local_translation_enabled = True
        cfg.runtime.tts_output_mode = "subtitle_only"
        routes = AudioRouteConfig(
            meeting_in="remote-in",
            microphone_in="local-in",
            speaker_out="speaker-out",
            meeting_out="meeting-out",
        )

        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            path = export_runtime_diagnostics(
                config_path="config.yaml",
                config=cfg,
                routes=routes,
                runtime_stats_text="router: running=True",
                recent_errors=["warn-1", "warn-2"],
            )

            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("mode: bidirectional", text)
            self.assertIn("meeting_in: remote-in", text)
            self.assertIn("microphone_in: local-in", text)
            self.assertIn("remote_translation_enabled: False", text)
            self.assertIn("local_translation_enabled: True", text)
            self.assertIn("asr_language_mode: auto", text)
            self.assertIn("tts_output_mode: subtitle_only", text)
            self.assertIn("router: running=True", text)
            self.assertIn("warn-2", text)

    def test_export_runtime_diagnostics_includes_overflow_and_latency_fields(self) -> None:
        cfg = AppConfig()
        cfg.runtime.max_pipeline_latency_ms = 2500
        cfg.runtime.display_partial_strategy = "stable_only"
        cfg.runtime.llm_queue_maxsize_local = 16
        cfg.runtime.llm_queue_maxsize_remote = 24
        routes = AudioRouteConfig(
            meeting_in="remote-in",
            microphone_in="local-in",
            speaker_out="speaker-out",
            meeting_out="meeting-out",
        )
        router_stats = {
            "translation_overflow": {"local": 3, "remote": 7},
            "latency": [
                {"source": "remote", "utterance_id": "u1", "speech_end_to_asr_final_ms": 400},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            path = export_runtime_diagnostics(
                config_path="config.yaml",
                config=cfg,
                routes=routes,
                runtime_stats_text="router: running=True",
                recent_errors=[],
                router_stats=router_stats,
            )

            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("max_pipeline_latency_ms: 2500", text)
            self.assertIn("display_partial_strategy: stable_only", text)
            self.assertIn("translation_overflow_local: 3", text)
            self.assertIn("translation_overflow_remote: 7", text)
            self.assertIn("u1", text)

    def test_export_session_report_contains_runtime_snapshot_and_recent_errors(self) -> None:
        cfg = AppConfig()
        cfg.runtime.remote_translation_enabled = True
        cfg.runtime.local_translation_enabled = False
        cfg.runtime.tts_output_mode = "tts"
        routes = AudioRouteConfig(
            meeting_in="remote-in",
            microphone_in="local-in",
            speaker_out="speaker-out",
            meeting_out="meeting-out",
        )
        payload = {
            "stats_before_stop": {
                "router": {"running": True},
                "translation_overflow": {"local": 2, "remote": 4},
                "latency": [{"source": "remote", "utterance_id": "u1"}],
            },
            "session_meta": {"duration_sec": 12.5},
        }

        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            path = export_session_report(
                config_path="config.yaml",
                config=cfg,
                routes=routes,
                payload=payload,
                recent_errors=["err-a", "err-b"],
            )

            self.assertTrue(path.exists())
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_devices"]["meeting_in"], "remote-in")
            self.assertEqual(report["selected_devices"]["speaker_out"], "speaker-out")
            self.assertTrue(report["runtime"]["remote_translation_enabled"])
            self.assertFalse(report["runtime"]["local_translation_enabled"])
            self.assertEqual(report["runtime"]["asr_language_mode"], "auto")
            self.assertEqual(report["runtime"]["tts_output_mode"], "tts")
            self.assertIn("max_pipeline_latency_ms", report["runtime"])
            self.assertIn("display_partial_strategy", report["runtime"])
            self.assertEqual(report["stats"], payload["stats_before_stop"])
            self.assertEqual(report["session_meta"], payload["session_meta"])
            self.assertEqual(report["recent_errors"][-1], "err-b")
            self.assertEqual(report["translation_overflow"], {"local": 2, "remote": 4})
            self.assertEqual(len(report["recent_latency"]), 1)


if __name__ == "__main__":
    unittest.main()
