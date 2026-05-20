#include "synctranslate_pcm_ring.h"

#define SYNCTRANSLATE_PCM_RING_POOLTAG 'RCSA'

typedef struct _SYNCTRANSLATE_PCM_RING
{
    KSPIN_LOCK lock;
    volatile LONG initState;
    FLOAT* samples;
    ULONG capacityFrames;
    ULONG readPos;
    ULONG writePos;
    ULONG bufferedFrames;
    ULONGLONG totalWrittenFrames;
    ULONGLONG totalReadFrames;
    ULONGLONG droppedFrames;
    ULONGLONG underrunFrames;
} SYNCTRANSLATE_PCM_RING;

static SYNCTRANSLATE_PCM_RING g_syncTranslatePcmRing = { 0 };

NTSTATUS
SyncTranslatePcmRingInitialize(
    _In_ ULONG capacityFrames
)
{
    PAGED_CODE();

    if (capacityFrames == 0)
    {
        return STATUS_INVALID_PARAMETER;
    }

    LONG previousState = InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 1, 0);
    if (previousState == 2)
    {
        return STATUS_SUCCESS;
    }

    if (previousState != 0)
    {
        return STATUS_DEVICE_BUSY;
    }

    SIZE_T allocationBytes = (SIZE_T)capacityFrames * sizeof(FLOAT);
    FLOAT* samples = (FLOAT*)ExAllocatePool2(
        POOL_FLAG_NON_PAGED,
        allocationBytes,
        SYNCTRANSLATE_PCM_RING_POOLTAG
    );
    if (samples == NULL)
    {
        InterlockedExchange(&g_syncTranslatePcmRing.initState, 0);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    RtlZeroMemory(samples, allocationBytes);
    KeInitializeSpinLock(&g_syncTranslatePcmRing.lock);
    g_syncTranslatePcmRing.samples = samples;
    g_syncTranslatePcmRing.capacityFrames = capacityFrames;
    g_syncTranslatePcmRing.readPos = 0;
    g_syncTranslatePcmRing.writePos = 0;
    g_syncTranslatePcmRing.bufferedFrames = 0;
    g_syncTranslatePcmRing.totalWrittenFrames = 0;
    g_syncTranslatePcmRing.totalReadFrames = 0;
    g_syncTranslatePcmRing.droppedFrames = 0;
    g_syncTranslatePcmRing.underrunFrames = 0;
    InterlockedExchange(&g_syncTranslatePcmRing.initState, 2);
    return STATUS_SUCCESS;
}

VOID
SyncTranslatePcmRingShutdown()
{
    FLOAT* samples = NULL;

    if (InterlockedExchange(&g_syncTranslatePcmRing.initState, 0) == 2)
    {
        samples = g_syncTranslatePcmRing.samples;
        g_syncTranslatePcmRing.samples = NULL;
        g_syncTranslatePcmRing.capacityFrames = 0;
        g_syncTranslatePcmRing.readPos = 0;
        g_syncTranslatePcmRing.writePos = 0;
        g_syncTranslatePcmRing.bufferedFrames = 0;
    }

    if (samples != NULL)
    {
        ExFreePoolWithTag(samples, SYNCTRANSLATE_PCM_RING_POOLTAG);
    }
}

VOID
SyncTranslatePcmRingFlush()
{
    if (InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return;
    }

    KIRQL oldIrql;
    KeAcquireSpinLock(&g_syncTranslatePcmRing.lock, &oldIrql);
    g_syncTranslatePcmRing.readPos = 0;
    g_syncTranslatePcmRing.writePos = 0;
    g_syncTranslatePcmRing.bufferedFrames = 0;
    KeReleaseSpinLock(&g_syncTranslatePcmRing.lock, oldIrql);
}

ULONG
SyncTranslatePcmRingWriteFloat32Mono(
    _In_reads_(frames) const FLOAT* samples,
    _In_ ULONG frames
)
{
    if (samples == NULL || frames == 0 ||
        InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return 0;
    }

    KIRQL oldIrql;
    KeAcquireSpinLock(&g_syncTranslatePcmRing.lock, &oldIrql);

    for (ULONG frame = 0; frame < frames; ++frame)
    {
        if (g_syncTranslatePcmRing.bufferedFrames >= g_syncTranslatePcmRing.capacityFrames)
        {
            g_syncTranslatePcmRing.readPos = (g_syncTranslatePcmRing.readPos + 1) % g_syncTranslatePcmRing.capacityFrames;
            g_syncTranslatePcmRing.bufferedFrames--;
            g_syncTranslatePcmRing.droppedFrames++;
        }

        g_syncTranslatePcmRing.samples[g_syncTranslatePcmRing.writePos] = samples[frame];
        g_syncTranslatePcmRing.writePos = (g_syncTranslatePcmRing.writePos + 1) % g_syncTranslatePcmRing.capacityFrames;
        g_syncTranslatePcmRing.bufferedFrames++;
        g_syncTranslatePcmRing.totalWrittenFrames++;
    }

    KeReleaseSpinLock(&g_syncTranslatePcmRing.lock, oldIrql);
    return frames;
}

ULONG
SyncTranslatePcmRingReadFloat32Mono(
    _Out_writes_(frames) FLOAT* samples,
    _In_ ULONG frames
)
{
    if (samples == NULL || frames == 0)
    {
        return 0;
    }

    RtlZeroMemory(samples, (SIZE_T)frames * sizeof(FLOAT));
    if (InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return 0;
    }

    ULONG framesRead = 0;
    KIRQL oldIrql;
    KeAcquireSpinLock(&g_syncTranslatePcmRing.lock, &oldIrql);

    for (ULONG frame = 0; frame < frames; ++frame)
    {
        if (g_syncTranslatePcmRing.bufferedFrames == 0)
        {
            g_syncTranslatePcmRing.underrunFrames += (frames - frame);
            break;
        }

        samples[frame] = g_syncTranslatePcmRing.samples[g_syncTranslatePcmRing.readPos];
        g_syncTranslatePcmRing.readPos = (g_syncTranslatePcmRing.readPos + 1) % g_syncTranslatePcmRing.capacityFrames;
        g_syncTranslatePcmRing.bufferedFrames--;
        g_syncTranslatePcmRing.totalReadFrames++;
        framesRead++;
    }

    KeReleaseSpinLock(&g_syncTranslatePcmRing.lock, oldIrql);
    return framesRead;
}

static ULONG
SyncTranslatePcmRingBytesPerSample(
    _In_ const WAVEFORMATEXTENSIBLE* wfExt
)
{
    if (wfExt == NULL || wfExt->Format.nBlockAlign == 0)
    {
        return 0;
    }

    ULONG channels = max(1u, wfExt->Format.nChannels);
    return wfExt->Format.nBlockAlign / channels;
}

static VOID
SyncTranslatePcmRingWriteSampleFromFloat(
    _Out_writes_bytes_(sampleBytes) BYTE* dst,
    _In_ ULONG sampleBytes,
    _In_ FLOAT value,
    _In_ USHORT formatTag,
    _In_opt_ const GUID* subFormat
)
{
    if (sampleBytes == sizeof(FLOAT) &&
        (formatTag == WAVE_FORMAT_IEEE_FLOAT ||
            (formatTag == WAVE_FORMAT_EXTENSIBLE &&
                subFormat != NULL &&
                IsEqualGUID(*subFormat, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT))))
    {
        *(FLOAT*)dst = value;
        return;
    }

    if (sampleBytes == sizeof(SHORT))
    {
        FLOAT clamped = max(-1.0f, min(1.0f, value));
        LONG scaled = (LONG)(clamped * 32767.0f);
        if (scaled > 32767)
        {
            scaled = 32767;
        }
        if (scaled < -32768)
        {
            scaled = -32768;
        }
        *(SHORT*)dst = (SHORT)scaled;
        return;
    }

    RtlZeroMemory(dst, sampleBytes);
}

VOID
SyncTranslatePcmRingReadIntoDma(
    _In_ const WAVEFORMATEXTENSIBLE* wfExt,
    _Out_writes_bytes_(bytes) BYTE* dst,
    _In_ ULONG bytes
)
{
    if (wfExt == NULL || dst == NULL || bytes == 0)
    {
        return;
    }

    RtlZeroMemory(dst, bytes);

    ULONG channels = max(1u, wfExt->Format.nChannels);
    ULONG sampleBytes = SyncTranslatePcmRingBytesPerSample(wfExt);
    if (sampleBytes == 0)
    {
        return;
    }

    ULONG bytesPerFrame = sampleBytes * channels;
    if (bytesPerFrame == 0)
    {
        return;
    }

    ULONG frames = bytes / bytesPerFrame;
    if (frames == 0)
    {
        return;
    }

    USHORT formatTag = wfExt->Format.wFormatTag;
    const GUID* subFormat = (formatTag == WAVE_FORMAT_EXTENSIBLE) ? &wfExt->SubFormat : NULL;

    for (ULONG frame = 0; frame < frames; ++frame)
    {
        FLOAT sample = 0.0f;
        (void)SyncTranslatePcmRingReadFloat32Mono(&sample, 1);

        BYTE* framePtr = dst + ((SIZE_T)frame * bytesPerFrame);
        for (ULONG channel = 0; channel < channels; ++channel)
        {
            SyncTranslatePcmRingWriteSampleFromFloat(
                framePtr + ((SIZE_T)channel * sampleBytes),
                sampleBytes,
                sample,
                formatTag,
                subFormat
            );
        }
    }
}

VOID
SyncTranslatePcmRingGetStats(
    _Out_ SYNCTRANSLATE_PCM_RING_STATS* stats
)
{
    if (stats == NULL)
    {
        return;
    }

    RtlZeroMemory(stats, sizeof(*stats));
    if (InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return;
    }

    KIRQL oldIrql;
    KeAcquireSpinLock(&g_syncTranslatePcmRing.lock, &oldIrql);
    stats->capacityFrames = g_syncTranslatePcmRing.capacityFrames;
    stats->bufferedFrames = g_syncTranslatePcmRing.bufferedFrames;
    stats->totalWrittenFrames = g_syncTranslatePcmRing.totalWrittenFrames;
    stats->totalReadFrames = g_syncTranslatePcmRing.totalReadFrames;
    stats->droppedFrames = g_syncTranslatePcmRing.droppedFrames;
    stats->underrunFrames = g_syncTranslatePcmRing.underrunFrames;
    KeReleaseSpinLock(&g_syncTranslatePcmRing.lock, oldIrql);
}
