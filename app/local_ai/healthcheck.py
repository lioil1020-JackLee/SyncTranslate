from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.infra.translation.provider import TranslationProvider
from app.infra.tts.edge_tts_adapter import EdgeTtsProvider


class AsrHealthCheckProvider(Protocol):
    def health_check(self) -> tuple[bool, str]:
        ...


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
    asr_engine: AsrHealthCheckProvider,
    llm_client: TranslationProvider,
    tts_engine: object,
) -> LocalHealthReport:
    asr_ok = True
    llm_ok = True
    tts_ok = True
    asr_message = "ready"
    llm_message = "ready"
    tts_message = "ready"

    try:
        asr_ok, asr_message = asr_engine.health_check()
    except Exception as exc:
        asr_ok = False
        asr_message = str(exc)

    try:
        llm_ok, llm_message = llm_client.health_check()
    except BaseException as exc:
        llm_ok = False
        llm_message = _format_healthcheck_exception(exc)

    if isinstance(tts_engine, EdgeTtsProvider):
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


def _format_healthcheck_exception(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted while waiting for local LLM response"
    text = str(exc).strip()
    return text or exc.__class__.__name__
