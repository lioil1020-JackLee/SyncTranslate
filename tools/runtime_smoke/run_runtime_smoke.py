from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time


@dataclass(slots=True)
class SmokeResult:
    name: str
    ok: bool
    returncode: int
    elapsed_sec: float
    command: list[str]
    stdout_tail: str
    stderr_tail: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tail(text: str, *, limit: int = 2000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def _run(name: str, command: list[str], *, timeout: float, root: Path) -> SmokeResult:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    return SmokeResult(
        name=name,
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        elapsed_sec=round(elapsed, 3),
        command=command,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _packaged_python_path(onedir: Path) -> Path:
    return onedir / "runtimes" / "shared" / "Scripts" / "python.exe"


def _packaged_exe_path(onedir: Path) -> Path:
    return onedir / "SyncTranslate.exe"


def _check_packaged_onedir(onedir: Path, *, root: Path) -> SmokeResult:
    required = [
        onedir,
        _packaged_python_path(onedir),
        onedir / "runtimes" / "shared",
        onedir / "runtimes" / "faster_whisper",
        onedir / "runtimes" / "models",
    ]
    missing = [str(path) for path in required if not path.exists()]
    exe_exists = _packaged_exe_path(onedir).exists()
    ok = not missing and exe_exists
    stdout = "packaged onedir structure OK" if ok else ""
    stderr_parts: list[str] = []
    if missing:
        stderr_parts.append("missing: " + ", ".join(missing))
    if not exe_exists:
        stderr_parts.append(f"missing: {onedir / 'SyncTranslate.exe'}")
    return SmokeResult(
        name="packaged_onedir_structure",
        ok=ok,
        returncode=0 if ok else 1,
        elapsed_sec=0.0,
        command=["check-packaged-onedir", str(onedir)],
        stdout_tail=stdout,
        stderr_tail="\n".join(stderr_parts),
    )


def _streaming_commands(root: Path, config: str, language: str) -> list[tuple[str, list[str], float]]:
    py = sys.executable
    commands: list[tuple[str, list[str], float]] = []
    if language in {"zh-TW", "all"}:
        audio = next((root / "downloads" / "benchmark" / "zh-TW").glob("*.wav"), None)
        ref = root / "downloads" / "benchmark_results" / "retest_20260416" / "chinese_reference_from_vtt.txt"
        if audio and ref.exists():
            commands.append(
                (
                    "streaming_replay_zh",
                    [
                        py,
                        "tools/asr_benchmark/streaming_sim.py",
                        "--config",
                        config,
                        "--audio",
                        str(audio),
                        "--reference",
                        str(ref),
                        "--source",
                        "local",
                        "--language",
                        "zh-TW",
                        "--speed",
                        "8",
                        "--output",
                        "downloads/benchmark_results/runtime_smoke_zh",
                    ],
                    300.0,
                )
            )
    if language in {"en", "all"}:
        audio = next((root / "downloads" / "benchmark" / "en").glob("*.wav"), None)
        ref = root / "downloads" / "benchmark_results" / "retest_20260416" / "english_reference_from_srt.txt"
        if audio and ref.exists():
            commands.append(
                (
                    "streaming_replay_en",
                    [
                        py,
                        "tools/asr_benchmark/streaming_sim.py",
                        "--config",
                        config,
                        "--audio",
                        str(audio),
                        "--reference",
                        str(ref),
                        "--source",
                        "remote",
                        "--language",
                        "en",
                        "--speed",
                        "8",
                        "--output",
                        "downloads/benchmark_results/runtime_smoke_en",
                    ],
                    240.0,
                )
            )
    return commands


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run runtime-level smoke checks that mimic real app entrypoints.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--with-pytest", action="store_true", help="Also run the pytest suite.")
    parser.add_argument(
        "--packaged-onedir",
        default="",
        help="Validate a PyInstaller onedir build and run --check through its bundled Python.",
    )
    parser.add_argument(
        "--with-streaming-replay",
        choices=["zh-TW", "en", "all"],
        default="",
        help="Also run benchmark audio through the streaming simulator.",
    )
    parser.add_argument("--report", default="logs/runtime_smoke_report.json")
    args = parser.parse_args(argv)

    root = _project_root()
    packaged_onedir = Path(args.packaged_onedir).resolve() if args.packaged_onedir else None
    if packaged_onedir is not None:
        packaged_exe = str(_packaged_exe_path(packaged_onedir))
        config_path = str((root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
        checks: list[tuple[str, list[str], float]] = [
            ("packaged_cli_config_check", [packaged_exe, "--config", config_path, "--check"], 120.0),
            (
                "packaged_healthcheck_worker",
                [packaged_exe, "--healthcheck-worker", "--healthcheck-config", config_path],
                180.0,
            ),
        ]
    else:
        py = sys.executable
        checks = [
            ("cli_config_check", [py, "main.py", "--check"], 120.0),
            ("healthcheck_worker", [py, "main.py", "--healthcheck-worker", "--healthcheck-config", args.config], 120.0),
            (
                "ui_healthcheck_path",
                [py, "tools/runtime_smoke/healthcheck_ui_smoke.py", "--config", args.config, "--timeout", "90"],
                150.0,
            ),
        ]
    if args.with_pytest:
        checks.append(("pytest", [py, "-m", "pytest", "-q"], 600.0))
    if args.with_streaming_replay:
        checks.extend(_streaming_commands(root, args.config, args.with_streaming_replay))

    results: list[SmokeResult] = []
    if packaged_onedir is not None:
        result = _check_packaged_onedir(packaged_onedir, root=root)
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        print(f"[runtime-smoke] {result.name}: {status}")
        if result.stdout_tail:
            print(result.stdout_tail)
        if result.stderr_tail:
            print(result.stderr_tail)
        if not result.ok:
            checks = []
    for name, command, timeout in checks:
        print(f"[runtime-smoke] {name} ...", flush=True)
        try:
            result = _run(name, command, timeout=timeout, root=root)
        except subprocess.TimeoutExpired as exc:
            result = SmokeResult(
                name=name,
                ok=False,
                returncode=124,
                elapsed_sec=timeout,
                command=command,
                stdout_tail=_tail(exc.stdout or ""),
                stderr_tail=f"timeout after {timeout} sec\n{_tail(exc.stderr or '')}".strip(),
            )
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        print(f"[runtime-smoke] {name}: {status} ({result.elapsed_sec}s)")
        if result.stdout_tail:
            print(result.stdout_tail)
        if result.stderr_tail:
            print(result.stderr_tail)

    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[runtime-smoke] report: {report_path}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
