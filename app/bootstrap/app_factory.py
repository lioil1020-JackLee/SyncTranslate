from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from PySide6.QtGui import QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication

from app.application.auto_populate_devices import AutoPopulateDevicesService
from app.application.first_run_readiness import evaluate_first_run_readiness
from app.domain.version import build_metadata
from app.infra.audio.device_registry import DeviceManager
from app.infra.config.schema import translation_enabled_for_source
from app.infra.config.settings_store import load_config, save_config
from app.infra.translation.engine import TranslatorManager
from app.infra.translation.provider import create_translation_provider
from app.ui.main_window import MainWindow
from tools.validation.common import build_check_report


def _default_config_path() -> str:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        target = (base_dir / "config.yaml").resolve()
        if target.exists():
            return str(target)
        bundled_candidates = []
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            bundled_candidates.append((Path(meipass) / "config.yaml").resolve())
        bundled_candidates.append((base_dir / "_internal" / "config.yaml").resolve())
        for candidate in bundled_candidates:
            if not candidate.exists():
                continue
            try:
                shutil.copyfile(candidate, target)
                return str(target)
            except Exception:
                return str(candidate)
    else:
        base_dir = Path(__file__).resolve().parents[2]
    return str((base_dir / "config.yaml").resolve())


def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_default_config_path())
    parser.add_argument("--check", action="store_true", help="Run config + device checks without opening the UI")
    parser.add_argument("--healthcheck-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--healthcheck-config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--llm-runtime-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--translation-smoke", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_from_cli(argv: list[str] | None = None) -> int:
    parser = create_cli_parser()
    args = parser.parse_args(argv)

    if args.healthcheck_worker:
        worker_args: list[str] = []
        if args.healthcheck_config:
            worker_args.append(args.healthcheck_config)
        from app.local_ai.healthcheck_worker import main as healthcheck_worker_main

        return int(healthcheck_worker_main(worker_args))

    config = load_config(args.config)
    
    # Auto-detect and persist virtual device names to ensure config stays in sync with system state.
    # This handles driver updates or reinstalls that may change virtual device names.
    try:
        populator = AutoPopulateDevicesService(exclude_virtual_devices=True)
        populator.populate(config.audio)
        save_config(config, args.config)
    except Exception:
        # If auto-populate fails, continue with existing config (graceful degradation)
        pass
    
    if args.llm_runtime_check:
        return _run_llm_runtime_check(config)
    if args.translation_smoke:
        return _run_translation_smoke(config)

    devices = DeviceManager().list_all()
    if args.check:
        report = build_check_report(args.config, probe_bridge=True)
        readiness = evaluate_first_run_readiness(config, probe_bridge=True)
        metadata = build_metadata(
            config_schema_version=int(getattr(config.runtime, "config_schema_version", 0) or 0),
            runtime_mode=str(getattr(config.runtime, "session_mode", "meeting") or "meeting"),
        )
        details = report.details
        bridge = next((item for item in report.items if item.name == "bridge_status"), None)
        driver = next((item for item in report.items if item.name == "driver_status"), None)
        audio_devices = next((item for item in report.items if item.name == "audio_devices"), None)
        device_details = audio_devices.details if audio_devices else {}
        print(f"Config OK: {args.config}")
        print(f"app_version={metadata.app_version}")
        print(f"git_commit={metadata.git_commit}")
        print(f"build_timestamp={metadata.build_timestamp}")
        print(f"packaged={str(metadata.packaged).lower()}")
        print(f"config_schema_version={details.get('config_schema_version', getattr(config.runtime, 'config_schema_version', 0))}")
        print(f"session_mode={details.get('session_mode', str(getattr(config.runtime, 'session_mode', 'meeting') or 'meeting'))}")
        print(f"meeting_audio_source={details.get('meeting_audio_source', str(getattr(config.meeting, 'audio_source', 'system_input') or 'system_input'))}")
        print(f"legacy_direction_mode={config.direction.mode}")
        print(f"audio_routing_mode={config.audio.routing_mode}")
        print(
            "runtime_modes="
            f"remote_translation_enabled={translation_enabled_for_source(config.runtime, 'remote')} "
            f"local_translation_enabled={translation_enabled_for_source(config.runtime, 'local')} "
            "asr_language_mode=fixed "
            f"tts_output_mode={str(getattr(config.runtime, 'tts_output_mode', 'subtitle_only') or 'subtitle_only')}"
        )
        print("asr_language_mode=fixed")
        print(f"selected_asr_backend={details.get('selected_asr_backend', config.asr.engine)}")
        print(f"virtual_audio_required={str(bool(details.get('virtual_audio_required', False))).lower()}")
        print(f"bridge_required={str(bool(details.get('bridge_required', False))).lower()}")
        print(f"driver_format_expected={details.get('driver_format_expected', '48000Hz PCM16 2ch')}")
        print(f"driver_format_status={details.get('driver_format_status', 'UNKNOWN')}")
        print(f"wdk_environment_status={details.get('wdk_environment_status', 'UNKNOWN')}")
        print(f"driver_build_tools_available={str(bool(details.get('driver_build_tools_available', False))).lower()}")
        print(f"protocol_v2_ready={str(bool(details.get('protocol_v2_ready', False))).lower()}")
        print(f"pcm16_stereo_boundary_ready={str(bool(details.get('pcm16_stereo_boundary_ready', False))).lower()}")
        print(f"bridge_status={bridge.status if bridge else 'WARN'} {bridge.message if bridge else 'not checked'}")
        print(f"driver_status={driver.status if driver else 'WARN'} {driver.message if driver else 'not checked'}")
        print(f"input_device_count={int(device_details.get('input_count', 0) or 0)}")
        print(f"output_device_count={int(device_details.get('output_count', 0) or 0)}")
        print(f"config_migration_status={details.get('config_migration_status', 'current')}")
        warnings = list(details.get("validation_warnings", []) or [])
        print(f"validation_warnings={','.join(str(item) for item in warnings) if warnings else 'none'}")
        print(
            "readiness="
            f"meeting_ready={readiness.summary.get('meeting_ready')} "
            f"dialogue_ready={readiness.summary.get('dialogue_ready')} "
            f"asr_model_ready={readiness.summary.get('asr_model_ready')} "
            f"llm_model_ready={readiness.summary.get('llm_model_ready')} "
            f"bridge_ready={readiness.summary.get('bridge_ready')} "
            f"driver_ready={readiness.summary.get('driver_ready')}"
        )
        print(f"suggested_next_action={readiness.summary.get('suggested_next_action')}")
        print(f"sample_rate={config.runtime.sample_rate} chunk_ms={config.runtime.chunk_ms}")
        print(f"asr={config.asr.engine}:{config.asr.model} device={config.asr.device}")
        print(
            f"llm={config.llm.backend} model={config.llm.model} "
            f"ctx={config.llm.runtime.ctx_size} gpu_layers={config.llm.runtime.gpu_layers} threads={config.llm.runtime.threads}"
        )
        print(f"llm_model_path={config.llm.runtime.model_path}")
        print(
            "meeting_tts="
            f"{config.meeting_tts.engine} voice={config.meeting_tts.voice_name or config.meeting_tts.model_path}"
        )
        print(
            "local_tts="
            f"{config.local_tts.engine} voice={config.local_tts.voice_name or config.local_tts.model_path}"
        )
        print(f"devices_found={len(devices)}")
        return 0 if report.status in {"PASS", "WARN"} else 1

    _prewarm_translation_runtime(config)

    _apply_windows_app_id()
    app = QApplication(sys.argv)
    _apply_default_ui_font(app)
    app.setApplicationName("SyncTranslate")
    app.setApplicationDisplayName("SyncTranslate")
    app.setOrganizationName("lioil")
    app.setDesktopFileName("lioil.synctranslate")
    icon_file = _resolve_icon_path()
    if icon_file:
        app.setWindowIcon(QIcon(icon_file))
    window = MainWindow(args.config)
    if icon_file:
        window.setWindowIcon(QIcon(icon_file))
    window.show()
    if icon_file:
        try:
            app.processEvents()
            if window.windowHandle():
                window.windowHandle().setIcon(QIcon(icon_file))
        except Exception:
            pass
    try:
        return app.exec()
    except KeyboardInterrupt:
        return 130


def _run_llm_runtime_check(config) -> int:
    from app.bootstrap.external_runtime import configure_external_ai_runtime
    from app.infra.translation.inprocess_adapter import _find_llama_cpp, _resolve_model_path

    info = configure_external_ai_runtime()
    Llama = _find_llama_cpp()
    model_path = _resolve_model_path(config.llm.runtime.model_path)
    payload = {
        "ok": bool(model_path.exists()),
        "llama_class": f"{Llama.__module__}.{Llama.__name__}",
        "model_path": str(model_path),
        "site_packages": info.get("site_packages", []),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def _run_translation_smoke(config) -> int:
    from app.infra.translation.provider import create_translation_provider

    provider = create_translation_provider(config.llm)
    started = time.perf_counter()
    translated = provider.translate(
        "Hello, this is a packaged runtime translation test.",
        source_lang="en",
        target_lang="zh-TW",
        context=[],
        profile=config.llm.profiles.live_caption_fast,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    payload = {
        "ok": bool(translated.strip()),
        "elapsed_ms": elapsed_ms,
        "translation": translated,
        "debug": provider.debug_snapshot(),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def _prewarm_translation_runtime(config) -> None:
    runtime_cfg = getattr(config, "runtime", None)
    if runtime_cfg is not None and not bool(getattr(runtime_cfg, "warmup_on_start", True)):
        return
    local_enabled = bool(translation_enabled_for_source(getattr(config, "runtime", None), "local"))
    remote_enabled = bool(translation_enabled_for_source(getattr(config, "runtime", None), "remote"))
    if not (local_enabled or remote_enabled):
        return

    effective_local = TranslatorManager._effective_llm_config(config.llm, config.llm_channels.local)
    effective_remote = TranslatorManager._effective_llm_config(config.llm, config.llm_channels.remote)

    warmup_targets: list[object] = []
    if local_enabled:
        warmup_targets.append(effective_local)
    if remote_enabled:
        warmup_targets.append(effective_remote)

    seen: set[tuple[str, str, int, int, int, int]] = set()
    for llm_cfg in warmup_targets:
        backend = str(getattr(llm_cfg, "backend", "") or "").strip().lower()
        runtime = getattr(llm_cfg, "runtime", None)
        model_path = str(getattr(runtime, "model_path", "") or "")
        ctx_size = int(getattr(runtime, "ctx_size", 4096) or 4096)
        gpu_layers = int(getattr(runtime, "gpu_layers", 0) or 0)
        threads = int(getattr(runtime, "threads", 1) or 1)
        batch_size = int(getattr(runtime, "batch_size", 1) or 1)
        key = (backend, model_path, ctx_size, gpu_layers, threads, batch_size)
        if key in seen:
            continue
        seen.add(key)
        try:
            provider = create_translation_provider(llm_cfg)
            warmup = getattr(provider, "warmup", None)
            if callable(warmup):
                warmup()
        except Exception as exc:
            print(f"[startup] llm warmup skipped: {exc}", file=sys.stderr)


def _apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("lioil.synctranslate")
    except Exception:
        return


def _resolve_icon_path() -> str | None:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    else:
        base = os.path.abspath(".")
    candidate = os.path.join(base, "lioil.ico")
    if os.path.exists(candidate):
        return candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate2 = os.path.join(script_dir, "..", "..", "lioil.ico")
    candidate2 = os.path.normpath(candidate2)
    if os.path.exists(candidate2):
        return candidate2

    repo_root_candidate = Path(__file__).resolve().parent.parent.parent / "lioil.ico"
    if repo_root_candidate.exists():
        return str(repo_root_candidate)
    return None


def _apply_default_ui_font(app: QApplication) -> None:
    # Avoid Qt trying legacy bitmap fonts (e.g., Fixedsys) on some Windows setups.
    preferred_families = [
        "Microsoft JhengHei UI",
        "Microsoft JhengHei",
        "Segoe UI",
        "Arial",
    ]
    try:
        available = set(QFontDatabase.families())
        chosen = next((family for family in preferred_families if family in available), "")
        if chosen:
            app.setFont(QFont(chosen, 10))
    except Exception:
        return


__all__ = ["create_cli_parser", "run_from_cli"]
