from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.translation.provider import create_translation_provider
from app.infra.tts.engine import create_tts_engine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.healthcheck import run_local_healthcheck
from app.infra.config.settings_store import load_config


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    config_path = args[0] if args else "config.yaml"

    try:
        config = load_config(config_path)
        asr = FasterWhisperEngine(
            model=config.asr.model,
            device=config.asr.device,
            compute_type=config.asr.compute_type,
            beam_size=config.asr.beam_size,
            final_beam_size=config.asr.final_beam_size,
            condition_on_previous_text=config.asr.condition_on_previous_text,
            final_condition_on_previous_text=config.asr.final_condition_on_previous_text,
            language="",
        )
        llm = create_translation_provider(config.llm)
        tts = create_tts_engine(config.meeting_tts)
        report = run_local_healthcheck(asr_engine=asr, llm_client=llm, tts_engine=tts)
    except BaseException as exc:
        message = _format_worker_exception(exc)
        report = LocalHealthReport(
            asr_ok=False,
            llm_ok=False,
            tts_ok=False,
            asr_message=message,
            llm_message=message,
            tts_message=message,
        )
    sys.stdout.write(json.dumps(asdict(report), ensure_ascii=False))
    sys.stdout.flush()
    sys.stderr.flush()
    # faster-whisper / CUDA teardown can crash this short-lived subprocess on exit
    # after the health report has already been produced. Exit immediately after flush.
    os._exit(0)


def _format_worker_exception(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "system check interrupted"
    text = str(exc).strip()
    return text or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
