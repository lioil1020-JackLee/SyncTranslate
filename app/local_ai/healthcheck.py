from __future__ import annotations

from dataclasses import dataclass

from app.local_ai.faster_whisper_engine import FasterWhisperEngine
from app.local_ai.ollama_client import OllamaClient
from app.local_ai.piper_tts import PiperTtsEngine
from app.model_providers import EdgeTtsProvider
from app.local_ai.runtime_paths import path_exists


@dataclass(slots=True)
class LocalHealthReport:
    asr_ok: bool
    llm_ok: bool
    tts_ok: bool
    asr_message: str
    llm_message: str
    tts_message: str

    @property
    def ok(self) -> bool:
        return self.asr_ok and self.llm_ok and self.tts_ok


def run_local_healthcheck(
    *,
    asr_engine: FasterWhisperEngine,
    llm_client: OllamaClient,
    tts_engine: object,
    warmup: bool = False,
) -> LocalHealthReport:
    asr_ok = True
    llm_ok = True
    tts_ok = True
    asr_message = "ready"
    llm_message = "ready"
    tts_message = "ready"

    try:
        if warmup:
            asr_engine.warmup()
        asr_ok, asr_message = asr_engine.health_check()
    except Exception as exc:
        asr_ok = False
        asr_message = str(exc)

    llm_ok, llm_message = llm_client.health_check()

    if isinstance(tts_engine, PiperTtsEngine):
        exe_ok = path_exists(tts_engine.executable_path)
        model_ok = path_exists(tts_engine.model_path)
        if not exe_ok or not model_ok:
            tts_ok = False
            tts_message = "piper executable/model not found"
    elif isinstance(tts_engine, EdgeTtsProvider):
        try:
            import edge_tts  # type: ignore  # noqa: F401
        except Exception:
            tts_ok = False
            tts_message = "edge-tts not installed"

    return LocalHealthReport(
        asr_ok=asr_ok,
        llm_ok=llm_ok,
        tts_ok=tts_ok,
        asr_message=asr_message,
        llm_message=llm_message,
        tts_message=tts_message,
    )
