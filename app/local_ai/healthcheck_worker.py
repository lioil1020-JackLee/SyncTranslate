from __future__ import annotations

import json
import sys
from dataclasses import asdict

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.healthcheck import run_local_healthcheck
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.tts_factory import create_tts_engine
from app.settings import load_config


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    config_path = args[0] if args else "config.yaml"
    warmup = len(args) > 1 and args[1].strip().lower() in {"1", "true", "yes", "on"}

    config = load_config(config_path)
    asr = FasterWhisperEngine(
        model=config.asr.model,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        beam_size=config.asr.beam_size,
        condition_on_previous_text=config.asr.condition_on_previous_text,
        language=config.language.meeting_source,
    )
    llm = OllamaClient(
        backend=config.llm.backend,
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        top_p=config.llm.top_p,
        request_timeout_sec=config.llm.request_timeout_sec,
    )
    tts = create_tts_engine(config.meeting_tts)
    report = run_local_healthcheck(asr_engine=asr, llm_client=llm, tts_engine=tts, warmup=warmup)
    print(json.dumps(asdict(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
