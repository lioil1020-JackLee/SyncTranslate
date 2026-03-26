# Architecture

## Runtime flow

SyncTranslate processes both directions at the same time:

1. `remote` input captures meeting audio.
2. `local` input captures microphone audio.
3. Each source goes through its own ASR channel.
4. Each source goes through its own LLM translation channel.
5. Captions are written into four panels:
   - remote original
   - remote translated
   - local original
   - local translated
6. Final translated speech can be sent to TTS on the opposite side.
7. `passthrough` forwards raw audio, but it no longer disables ASR.

## Package layout

- `app/application/`: orchestration services and session control
- `app/bootstrap/`: CLI startup, dependency wiring, runtime path helpers
- `app/domain/`: runtime state and transcript domain models
- `app/infra/asr/`: Faster-Whisper adapters and streaming workers
- `app/infra/audio/`: device discovery, capture, playback, routing helpers
- `app/infra/config/`: schema, migration, persistence
- `app/infra/translation/`: LM Studio translation providers and stitching
- `app/infra/tts/`: edge-tts playback queue and voice policy
- `app/local_ai/`: system-check worker process
- `app/ui/`: main window, pages, debug widgets
- `tests/`: automated regression coverage

## Direction model

The application no longer exposes old session modes such as:

- remote only
- local only
- remote -> local
- local -> remote

The project now assumes two independent audio lines and always runs in bidirectional mode.

## Model strategy

ASR and LLM settings are direction-specific by design:

- `asr_channels.local` and `asr_channels.remote`
- `llm_channels.local` and `llm_channels.remote`

This is the source of truth for runtime behavior.

## Health model

The old warmup / preheat path was removed.

The app now exposes one action only:

- system check

The system check validates:

- ASR availability
- LM Studio connectivity
- edge-tts availability
