#pragma once

#include <sysvad.h>

#define SYNCTRANSLATE_PCM_RING_DEFAULT_FRAMES (48000u * 5u)

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
SyncTranslatePcmRingWriteFloat32Mono(
    _In_reads_(frames) const FLOAT* samples,
    _In_ ULONG frames
);

ULONG
SyncTranslatePcmRingReadFloat32Mono(
    _Out_writes_(frames) FLOAT* samples,
    _In_ ULONG frames
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
