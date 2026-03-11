# SyncTranslate

Desktop interpreter assistant for meetings.

It captures meeting audio and microphone audio, runs local ASR + local LLM translation, and plays translated TTS to different output devices.

## Features

- Dual pipeline: `meeting -> local speaker` and `local mic -> meeting output`
- Local ASR with `faster-whisper`
- Local translation with `Ollama` or `LM Studio`
- TTS per direction
- `Piper` for offline local TTS
- `Edge TTS` for selectable male/female cloud voices
- Live captions with auto-scroll
- Audio route diagnostics and output tests

## Requirements

- Windows
- Python `3.11+`
- `uv`
- Optional GPU for faster ASR

## Install

```powershell
uv sync --extra local
```

## Run

```powershell
uv run python .\main.py
```

## Quick Check

```powershell
uv run python .\main.py --check
```

## Config Files

- Main local config: `config.yaml`
- Example config: `config.example.yaml`

`config.yaml` is intentionally ignored by git because it contains machine-specific audio routes and local runtime settings.

## TTS Notes

- `meeting_tts`: played to your local speaker/headphones
- `local_tts`: played to the meeting output device
- For output-device testing, the app now prefers local playable TTS and falls back to a local tone if the configured cloud TTS fails

## Common Workflow

1. Open `音訊路由與診斷`
2. Select input/output devices
3. Open `本地 AI`
4. Choose ASR / LLM / TTS settings
5. Run health check in diagnostics
6. Start the session from `即時字幕`

## Project Files Not Meant For Git

These are ignored:

- local virtual envs and uv cache
- downloaded models and tools
- machine-specific `config.yaml`
- exported diagnostics / logs

