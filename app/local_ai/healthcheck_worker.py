from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.healthcheck import run_local_healthcheck
from app.local_ai.llm_provider import create_translation_provider
from app.local_ai.tts_factory import create_tts_engine
from app.settings import load_config


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    config_path = args[0] if args else "config.yaml"
    warmup = len(args) > 1 and args[1].strip().lower() in {"1", "true", "yes", "on"}

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
            language=config.language.meeting_source,
        )
        llm = create_translation_provider(config.llm)
        tts = create_tts_engine(config.meeting_tts)
        report = run_local_healthcheck(asr_engine=asr, llm_client=llm, tts_engine=tts, warmup=warmup)
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
        return "health check interrupted"
    text = str(exc).strip()
    return text or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
