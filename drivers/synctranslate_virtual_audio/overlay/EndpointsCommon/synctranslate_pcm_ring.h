#pragma once

#include <sysvad.h>

#define SYNCTRANSLATE_VIRTUAL_AUDIO_SAMPLE_RATE 48000u
#define SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS 2u
#define SYNCTRANSLATE_VIRTUAL_AUDIO_BITS_PER_SAMPLE 16u
#define SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_SAMPLE 2u
#define SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME \
    (SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS * SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_SAMPLE)

#define SYNCTRANSLATE_PCM_RING_DEFAULT_FRAMES (SYNCTRANSLATE_VIRTUAL_AUDIO_SAMPLE_RATE * 5u)

typedef struct _SYNCTRANSLATE_PCM_RING_STATS
{
    ULONGLONG capacityFrames;
    ULONGLONG bufferedFrames;
    ULONGLONG totalWrittenFrames;
    ULONGLONG totalReadFrames;
    ULONGLONG droppedFrames;
    ULONGLONG underrunFrames;
} SYNCTRANSLATE_PCM_RING_STATS;

NTSTATUS
SyncTranslatePcmRingInitialize(
    _In_ ULONG capacityFrames
);

VOID
SyncTranslatePcmRingShutdown();

VOID
SyncTranslatePcmRingFlush();

ULONG
SyncTranslatePcmRingWritePcm16Stereo(
    _In_reads_bytes_(bytes) const BYTE* pcmBytes,
    _In_ ULONG bytes
);

ULONG
SyncTranslatePcmRingReadPcm16Stereo(
    _Out_writes_bytes_(bytes) BYTE* pcmBytes,
    _In_ ULONG bytes
);

VOID
SyncTranslatePcmRingReadIntoDma(
    _In_ const WAVEFORMATEXTENSIBLE* wfExt,
    _Out_writes_bytes_(bytes) BYTE* dst,
    _In_ ULONG bytes
);

VOID
SyncTranslatePcmRingGetStats(
    _Out_ SYNCTRANSLATE_PCM_RING_STATS* stats
);
