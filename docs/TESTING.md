# Testing

## Automated test commands

Run the full automated suite:

```powershell
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Run the non-UI config and device check:

```powershell
uv run python .\main.py --check
```

## Coverage focus

The current suite verifies:

- audio router policy
- bidirectional runtime behavior
- ASR auto-detect behavior
- per-direction ASR and LLM selection
- translation enable switches for remote and local
- edge-tts style policy
- transcript buffering
- session lifecycle
- config migration and canonical save format
- diagnostics export
- four-panel subtitle export
- UI behavior and labels
- end-to-end fake pipeline flow from ASR to translation to subtitles / TTS

## Manual smoke checklist

1. Start LM Studio and load the models configured for `llm_channels.local` and `llm_channels.remote`.
2. Open the app and confirm all four audio routes are selected.
3. Run system check and confirm ASR / LLM / TTS all report ready.
4. Start a live session.
5. Speak into the microphone and confirm:
   - local original updates
   - local translated updates
   - remote-side TTS follows the translated target language when enabled
6. Feed meeting audio and confirm:
   - remote original updates
   - remote translated updates
   - local-side TTS follows the translated target language when enabled
7. Switch `tts_output_mode` between `tts`, `subtitle_only`, and `passthrough`.
8. Export each subtitle panel and confirm the file content matches the UI.

## Expected steady-state rules

- Both sources remain active in bidirectional mode.
- Passthrough does not disable ASR.
- ASR source language stays auto-detect only.
- Remote and local translation targets remain independent.
- Remote and local translation enable switches remain independent.
