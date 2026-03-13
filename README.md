# SyncTranslate

SyncTranslate is a Windows desktop real-time interpretation tool for meetings.

It captures local/meeting audio, performs streaming ASR with faster-whisper, translates via LM Studio, and plays translated speech with Edge TTS. The GUI is built with PySide6 and provides routing, diagnostics, and live caption controls.

## Features

- Direction modes:
  - `meeting_to_local`
  - `local_to_meeting`
  - `bidirectional`
- Streaming ASR (`faster-whisper`) with VAD tuning
- Local translation backend (`lm_studio`)
- Edge TTS for both local and remote channels
- Live captions with partial/final event upsert by utterance revision
- Session lifecycle state machine (`idle/starting/running/stopping/failed`)
- Runtime diagnostics export and per-session report export

## Architecture (current)

- Entry and window:
  - `main.py`
  - `app/main_window.py`
  - `app/ui_main.py`
- Composition/runtime:
  - `app/app_bootstrap.py`
  - `app/runtime_facade.py`
  - `app/config_apply_service.py`
  - `app/diagnostics_service.py`
- Realtime pipeline:
  - `app/audio_router.py`
  - `app/audio_input_manager.py`
  - `app/asr_manager.py`
  - `app/translator_manager.py`
  - `app/tts_manager.py`
  - `app/state_manager.py`
  - `app/transcript_buffer.py`
- Local AI:
  - `app/local_ai/faster_whisper_engine.py`
  - `app/local_ai/streaming_asr.py`
  - `app/local_ai/translation_stitcher.py`
  - `app/local_ai/lm_studio_client.py`
  - `app/local_ai/healthcheck.py`

## Requirements

- Windows 10/11
- Python 3.11+
- `uv` (recommended)
- Audio devices available to `sounddevice`
- LM Studio running and model loaded
- For full local ASR, install optional dependency group `local`

## Quick Start (dev)

1. Install dependencies:

```powershell
uv sync --extra local --group dev
```

2. Prepare config (optional):

```powershell
# If missing, app will auto-create config.yaml from embedded defaults on first run.
```

3. Run GUI:

```powershell
uv run python .\main.py
```

4. Sanity check without GUI:

```powershell
uv run python .\main.py --check
```

## Configuration

Primary config is `config.yaml`. If missing, app auto-creates it from embedded defaults.

Main sections:

- `audio`: input/output devices and gain/volume
- `direction.mode`: routing mode
- `language`: source/target for local and meeting channels
- `asr`: model/device/compute/vad/streaming
- `llm`: LM Studio endpoint, model, profiles
- `tts`, `meeting_tts`, `local_tts`, `tts_channels`
- `runtime`: queue sizes, chunk size, echo guard, warmup

## Runtime Notes

- `ASRManager` includes pipeline revision + runtime fingerprint metadata on ASR events.
- `TranscriptBuffer` stores `channel`, `kind`, `utterance_id`, `revision`, and optional `latency_ms`.
- `AudioRouter` exposes event hooks (`asr_event`, `translation_event`, `tts_request`, `diagnostic_event`) and routes each stage explicitly.
- Session stop writes JSON report to `logs/session_reports`.

## Diagnostics

- Runtime event log: `logs/runtime_events.log`
- Health check (ASR/LLM/TTS) available from UI
- Export diagnostics text from UI

## Build Executables (PyInstaller via uv)

Spec files:

- `SyncTranslate-onedir.spec`
- `SyncTranslate-onefile.spec`

Build commands:

```powershell
uv run pyinstaller .\SyncTranslate-onedir.spec --noconfirm --clean
uv run pyinstaller .\SyncTranslate-onefile.spec --noconfirm --clean
```

Output:

- One-dir: `dist/SyncTranslate-onedir/`
- One-file: `dist/SyncTranslate.exe`

## Git Workflow (release)

Typical release flow:

```powershell
git add -A
git commit -m "release: v1.0.0"
git push
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

- No captions:
  - check input device routing and capture level
  - lower `asr.vad.rms_threshold`
- Captions but no audio:
  - verify `speaker_out` / `meeting_out`
  - run local TTS test in UI
- High translation latency:
  - reduce `llm.sliding_window.max_context_items`
  - use smaller LM model
- ASR too slow:
  - use smaller whisper model
  - prefer `cuda` and compatible compute type
