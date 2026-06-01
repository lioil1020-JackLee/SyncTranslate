from __future__ import annotations

import numpy as np

from app.infra.asr.manager_v2 import ASRManagerV2
from app.infra.config.schema import AppConfig
from tools.validation.meeting_source_stream_test import _metrics, create_parser


def test_meeting_source_stream_parser_supports_system_input_and_loopback() -> None:
    parser = create_parser()

    args = parser.parse_args(["--sources", "both", "--duration", "1.5", "--play-test-speech"])

    assert args.sources == "both"
    assert args.duration == 1.5
    assert args.play_test_speech is True
    assert args.sequential is False


def test_meeting_source_stream_metrics_warn_on_low_loopback_level() -> None:
    quiet = np.full((48000, 2), 0.01, dtype=np.float32)

    metrics = _metrics(quiet, 48000, [0.04, 0.04])

    assert metrics["level_status"] == "WARN"
    assert metrics["recommended_frontend_gain_to_0_05_rms"] == 5.0


def test_asr_manager_honors_configured_frontend_max_gain() -> None:
    cfg = AppConfig()
    cfg.runtime.asr_frontend_max_gain = 3.5

    manager = ASRManagerV2(cfg)

    assert manager._frontend_max_gain() == 3.5


def test_asr_manager_clamps_frontend_max_gain() -> None:
    cfg = AppConfig()
    cfg.runtime.asr_frontend_max_gain = 100.0

    manager = ASRManagerV2(cfg)

    assert manager._frontend_max_gain() == 8.0
