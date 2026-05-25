# Virtual Audio Protocol v2

Protocol v2 fixes the app/bridge/driver virtual audio boundary.

## Boundary

```text
sample_rate = 48000Hz
format      = PCM16
channels    = 2ch
layout      = interleaved stereo
frame_size  = 4 bytes
```

The bridge packet contains:

```json
{
  "protocol_version": 2,
  "sample_rate": 48000,
  "channels": 2,
  "bit_depth": 16,
  "dtype": "int16",
  "layout": "interleaved_stereo",
  "frames": 480,
  "data_b64": "..."
}
```

## Layering

- App internal audio remains float32 `AudioFrame`.
- ASR input remains 16000Hz mono float32 for faster-whisper/CTranslate2.
- Virtual audio driver/bridge boundary is 48000Hz PCM16 2ch.
- Direct passthrough must preserve stereo until the boundary conversion.

## Driver Contract

`IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM` accepts raw PCM16 interleaved stereo bytes. Payload length must be aligned to 4 bytes per stereo frame.

The kernel ring buffer counts buffered, written, read, dropped, and underrun values in stereo frames.

## Product Modes

meeting mode does not require the virtual audio driver or bridge.

dialogue mode requires SyncTranslate Virtual Speaker, SyncTranslate Virtual Microphone, and bridge readiness. If the driver format is not 48000Hz / PCM16 / 2ch, dialogue mode validation must fail or clearly warn when the format cannot be inspected.
