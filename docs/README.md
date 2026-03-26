# SyncTranslate

SyncTranslate is a Windows desktop live interpretation tool for two fully independent audio directions:

- Remote speech -> subtitles / translation / local TTS
- Local speech -> subtitles / translation / remote TTS

The current project is built around these rules:

- Audio routing is always bidirectional.
- ASR source language is always auto-detected.
- Remote and local translation targets are configured independently.
- Remote and local translation enable switches are independent.
- ASR and LLM models are always selected per direction.
- TTS uses `edge-tts`.
- Startup warmup was removed; the app now provides only a system check.

## Entry points

Run the desktop app:

```powershell
uv run python .\main.py
```

Run configuration and device check without opening the UI:

```powershell
uv run python .\main.py --check
```

## Project documents

- [Architecture](./ARCHITECTURE.md)
- [Configuration](./CONFIGURATION.md)
- [Testing](./TESTING.md)
- [Changelog](./CHANGELOG.md)

## Runtime notes

- Runtime crash logs and event logs are written to the system temp directory under `SyncTranslate`.
- Health-check snapshots are also written to the system temp directory, not the repository.
- The repository itself should stay clean after normal runs.
