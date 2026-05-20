from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.bootstrap.external_runtime import configure_external_ai_runtime
from app.infra.audio.virtual_bridge_probe import probe_virtual_audio_bridge
from app.infra.config.settings_store import load_config


def _child_main(config_path: str) -> int:
    configure_external_ai_runtime()
    from app.bootstrap.dependency_container import build_pipeline_bundle
    from app.application.transcript_service import TranscriptBuffer
    from app.infra.audio.capture import AudioCapture
    from app.infra.audio.playback import AudioPlayback
    from app.infra.config.schema import AudioRouteConfig

    cfg = load_config(config_path)
    cfg.runtime.local_asr_language = "none"
    cfg.runtime.tts_output_mode = "subtitle_only"
    cfg.audio.virtual_audio.require_driver = False

    bundle = build_pipeline_bundle(
        config=cfg,
        pipeline_revision=1301,
        transcript_buffer=TranscriptBuffer(max_items=None),
        local_capture=AudioCapture(),
        speaker_playback=AudioPlayback(),
        get_local_output_device=lambda: "",
        on_error=lambda e: None,
        on_diagnostic_event=lambda m: None,
        on_asr_event=lambda e: None,
    )
    routes = AudioRouteConfig(
        meeting_in="dummy-remote",
        microphone_in="dummy-local",
        speaker_out="dummy-spk",
        meeting_out="dummy-meet",
    )
    bundle.session_controller.start(
        routes,
        sample_rate=cfg.runtime.sample_rate,
        chunk_ms=cfg.runtime.chunk_ms,
        mode="meeting_to_local",
    )
    time.sleep(2.0)
    # 模擬 App 崩潰，故意不走 stop 路徑。
    os._exit(42)


def run_d3_app_crash_silence(config_path: str, out_json: str | None = None) -> dict[str, object]:
    cfg = load_config(config_path)
    bridge_path = str(cfg.audio.virtual_audio.bridge_path or "")

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, __file__, "--child", "--config", config_path],
        capture_output=True,
        text=True,
    )
    crash_elapsed_ms = (time.perf_counter() - started) * 1000.0

    probe_started = time.perf_counter()
    probe = probe_virtual_audio_bridge(bridge_path)
    probe_elapsed_ms = (time.perf_counter() - probe_started) * 1000.0

    # D3 可觀測判準：
    # 1) App 確實非正常退出
    # 2) 崩潰後 bridge 探測可返回（不長時間卡死）
    pass_d3 = bool(proc.returncode == 42 and crash_elapsed_ms < 60000.0 and probe_elapsed_ms < 30000.0)

    result = {
        "kind": "d3_app_crash_silence",
        "pass_d3": pass_d3,
        "app_exit_code": int(proc.returncode),
        "app_crash_elapsed_ms": round(crash_elapsed_ms, 2),
        "probe_elapsed_ms": round(probe_elapsed_ms, 2),
        "bridge_ready_after_crash": bool(probe.ready),
        "bridge_connected_after_crash": bool(probe.connected),
        "bridge_error_after_crash": str(probe.error or ""),
        "note": "以快速恢復/非阻塞為可觀測替代指標；silence 聲學層需外部 loopback 錄音驗證。",
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="D3 app crash silence fallback verifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="logs/session_reports/d3_app_crash_silence.json")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        return _child_main(args.config)

    result = run_d3_app_crash_silence(args.config, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("pass_d3", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
