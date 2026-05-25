# SyncTranslate Virtual Audio Format Contract v2

This document records the app/bridge/driver boundary format used by the v2
productized audio path.

## Fixed Virtual Endpoint Format

- Sample rate: 48000 Hz
- Sample format: PCM16 signed little-endian
- Channels: 2 channels
- Layout: interleaved stereo, frame order `L0, R0, L1, R1, ...`
- Bytes per sample: 2
- Bytes per stereo frame: 4

The kernel ring buffer counts capacity, buffered data, written data, read data,
drops, and underruns in stereo frames. Byte lengths on the WRITE_PCM IOCTL must
be aligned to 4-byte stereo frames.

## Layering

App internal audio remains float32 AudioFrame with sample-rate and channel
metadata. The Python bridge/client converts to PCM16 stereo 48k only at the
virtual audio boundary.

ASR input remains 16000 Hz mono float32 and is not part of this driver boundary.

## Meeting Mode

meeting mode does not require the driver, virtual bridge, virtual speaker, or
virtual microphone. Missing driver artifacts must not block meeting-mode
startup.

## Dialogue Mode

dialogue mode requires the virtual audio driver and bridge. Direct passthrough
will use this PCM16 stereo 48k boundary when writing to the virtual microphone
or reading from the virtual speaker.

## Current Source Mapping

- `overlay/EndpointsCommon/synctranslate_pcm_ring.h`
  defines the v2 constants and ring stats.
- `overlay/EndpointsCommon/synctranslate_pcm_ring.cpp`
  stores PCM16 interleaved stereo frames in NonPagedPool.
- `overlay/synctranslate_control.cpp`
  accepts only WRITE_PCM payloads aligned to 4-byte stereo frames.
- Upstream SysVAD endpoint format tables are applied by the driver build
  overlay process. WDK build validation must confirm both virtual speaker and
  virtual microphone enumerate as 48000 Hz / PCM16 / 2ch.
