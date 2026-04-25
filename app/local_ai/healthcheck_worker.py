from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from dataclasses import dataclass

from app.bootstrap.external_runtime import configure_external_ai_runtime

configure_external_ai_runtime()

from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.config.schema import AsrConfig
from app.infra.config.settings_store import load_config
from app.infra.asr.profile_selection import iter_active_asr_profiles_for_sources
from app.infra.translation.provider import create_translation_provider
from app.infra.tts.engine import create_tts_engine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.healthcheck import run_local_healthcheck


@dataclass(slots=True)
class _CombinedAsrHealthProbe:
    faster_whisper_probes: list[tuple[str, object]]

    def health_check(self) -> tuple[bool, str]:
        if not self.faster_whisper_probes:
            return False, "ASR probes not configured"

        all_ok = True
        details: list[str] = []
        for label, probe in self.faster_whisper_probes:
            try:
                ok, detail = probe.health_check()
            except Exception as exc:
                ok = False
                detail = str(exc)
            all_ok = all_ok and bool(ok)
            state = "ok" if ok else "failed"
            details.append(f"{label}: {state} - {detail}")
        return all_ok, "; ".join(details)


def _pick_channel_configs(config) -> list[tuple[str, str, AsrConfig]]:
    if bool(getattr(config.runtime, "use_channel_specific_asr", False)):
        profiles: list[tuple[str, str, AsrConfig]] = []
        for source, language, profile in iter_active_asr_profiles_for_sources(config):
            display_language = language or "auto"
            profiles.append((f"{source}/{display_language}", language, profile))
        return profiles
    return [("shared", "", config.asr)]


def _build_faster_whisper_probe(asr_cfg: AsrConfig, *, language: str = "") -> FasterWhisperEngine:
    return FasterWhisperEngine(
        model=asr_cfg.model,
        device=asr_cfg.device,
        compute_type=asr_cfg.compute_type,
        beam_size=asr_cfg.beam_size,
        final_beam_size=asr_cfg.final_beam_size,
        condition_on_previous_text=asr_cfg.condition_on_previous_text,
        final_condition_on_previous_text=asr_cfg.final_condition_on_previous_text,
        initial_prompt=asr_cfg.initial_prompt,
        hotwords=asr_cfg.hotwords,
        speculative_draft_model=asr_cfg.speculative_draft_model,
        speculative_num_beams=asr_cfg.speculative_num_beams,
        language=language,
    )


def _build_combined_asr_probe(config) -> _CombinedAsrHealthProbe:
    faster_whisper_probes: list[tuple[str, FasterWhisperEngine]] = []

    for label, language, asr_cfg in _pick_channel_configs(config):
        engine = str(getattr(asr_cfg, "engine", "") or "").strip().lower()
        if engine == "faster_whisper":
            faster_whisper_probes.append((label, _build_faster_whisper_probe(asr_cfg, language=language)))

    if not faster_whisper_probes:
        fallback_cfg = config.asr
        faster_whisper_probes.append(("shared", _build_faster_whisper_probe(fallback_cfg)))

    return _CombinedAsrHealthProbe(faster_whisper_probes=faster_whisper_probes)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    config_path = args[0] if args else "config.yaml"

    try:
        config = load_config(config_path)
        asr = _build_combined_asr_probe(config)
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
        return "健康檢查已中斷"
    text = str(exc).strip()
    return text or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
