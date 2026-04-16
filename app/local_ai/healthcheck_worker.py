from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from dataclasses import dataclass

from app.bootstrap.external_runtime import configure_external_ai_runtime

configure_external_ai_runtime()

from app.infra.asr.funasr_registry import get_funasr_registry
from app.infra.asr.faster_whisper_adapter import FasterWhisperEngine
from app.infra.config.schema import AsrConfig
from app.infra.translation.provider import create_translation_provider
from app.infra.tts.engine import create_tts_engine
from app.local_ai.healthcheck import LocalHealthReport
from app.local_ai.healthcheck import run_local_healthcheck
from app.infra.config.settings_store import load_config


@dataclass(slots=True)
class _FunAsrHealthProbe:
    model: str
    device: str

    def health_check(self) -> tuple[bool, str]:
        registry = get_funasr_registry()
        handle = registry.get_asr(model_name=self.model, requested_device=self.device)
        return True, f"{handle.device_effective} (funasr)"


@dataclass(slots=True)
class _CombinedAsrHealthProbe:
    funasr_probe: object | None
    faster_whisper_probe: object | None

    def health_check(self) -> tuple[bool, str]:
        checks: list[tuple[str, object]] = []
        if self.funasr_probe is not None:
            checks.append(("funasr", self.funasr_probe))
        if self.faster_whisper_probe is not None:
            checks.append(("faster-whisper", self.faster_whisper_probe))
        if not checks:
            return False, "ASR probes not configured"

        all_ok = True
        messages: list[str] = []
        for name, probe in checks:
            try:
                ok, detail = probe.health_check()
            except Exception as exc:
                ok = False
                detail = str(exc)
            all_ok = all_ok and bool(ok)
            state = "ok" if ok else "failed"
            messages.append(f"{name}: {state} - {detail}")
        return all_ok, " | ".join(messages)


def _pick_channel_configs(config) -> list[AsrConfig]:
    if bool(getattr(config.runtime, "use_channel_specific_asr", False)):
        remote = getattr(config.asr_channels, "remote", None)
        local = getattr(config.asr_channels, "local", None)
        ordered = [remote, local]
        return [item for item in ordered if item is not None]
    return [config.asr]


def _build_faster_whisper_probe(asr_cfg: AsrConfig) -> FasterWhisperEngine:
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
        language="",
    )


def _build_combined_asr_probe(config) -> _CombinedAsrHealthProbe:
    funasr_probe = None
    faster_whisper_probe = None

    for asr_cfg in _pick_channel_configs(config):
        engine = str(getattr(asr_cfg, "engine", "")).strip().lower()
        if engine == "funasr" and funasr_probe is None:
            model_name = str(getattr(asr_cfg.funasr, "model", "") or asr_cfg.model)
            funasr_probe = _FunAsrHealthProbe(model=model_name, device=asr_cfg.device)
        if engine == "faster_whisper" and faster_whisper_probe is None:
            faster_whisper_probe = _build_faster_whisper_probe(asr_cfg)

    if funasr_probe is None and faster_whisper_probe is None:
        # Backward compatibility for older single-ASR configs.
        fallback_cfg = config.asr
        fallback_engine = str(getattr(fallback_cfg, "engine", "faster_whisper")).strip().lower()
        if fallback_engine == "funasr":
            model_name = str(getattr(fallback_cfg.funasr, "model", "") or fallback_cfg.model)
            funasr_probe = _FunAsrHealthProbe(model=model_name, device=fallback_cfg.device)
        else:
            faster_whisper_probe = _build_faster_whisper_probe(fallback_cfg)

    return _CombinedAsrHealthProbe(
        funasr_probe=funasr_probe,
        faster_whisper_probe=faster_whisper_probe,
    )


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
        return "system check interrupted"
    text = str(exc).strip()
    return text or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
