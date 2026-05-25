#include "synctranslate_pcm_ring.h"

#define SYNCTRANSLATE_PCM_RING_POOLTAG 'RCSA'

typedef struct _SYNCTRANSLATE_PCM_RING
{
    KSPIN_LOCK lock;
    volatile LONG initState;
    SHORT* samples;
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

    SIZE_T allocationBytes = (SIZE_T)capacityFrames * SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME;
    SHORT* samples = (SHORT*)ExAllocatePool2(
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
    SHORT* samples = NULL;

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
SyncTranslatePcmRingWritePcm16Stereo(
    _In_reads_bytes_(bytes) const BYTE* pcmBytes,
    _In_ ULONG bytes
)
{
    if (pcmBytes == NULL || bytes == 0 ||
        (bytes % SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME) != 0 ||
        InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return 0;
    }

    ULONG frames = bytes / SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME;
    const SHORT* sourceSamples = (const SHORT*)pcmBytes;

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

        SIZE_T sourceIndex = (SIZE_T)frame * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS;
        SIZE_T ringIndex = (SIZE_T)g_syncTranslatePcmRing.writePos * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS;
        g_syncTranslatePcmRing.samples[ringIndex] = sourceSamples[sourceIndex];
        g_syncTranslatePcmRing.samples[ringIndex + 1] = sourceSamples[sourceIndex + 1];
        g_syncTranslatePcmRing.writePos = (g_syncTranslatePcmRing.writePos + 1) % g_syncTranslatePcmRing.capacityFrames;
        g_syncTranslatePcmRing.bufferedFrames++;
        g_syncTranslatePcmRing.totalWrittenFrames++;
    }

    KeReleaseSpinLock(&g_syncTranslatePcmRing.lock, oldIrql);
    return frames;
}

ULONG
SyncTranslatePcmRingReadPcm16Stereo(
    _Out_writes_bytes_(bytes) BYTE* pcmBytes,
    _In_ ULONG bytes
)
{
    if (pcmBytes == NULL || bytes == 0 ||
        (bytes % SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME) != 0)
    {
        return 0;
    }

    RtlZeroMemory(pcmBytes, bytes);
    if (InterlockedCompareExchange(&g_syncTranslatePcmRing.initState, 2, 2) != 2)
    {
        return 0;
    }

    ULONG frames = bytes / SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME;
    SHORT* destinationSamples = (SHORT*)pcmBytes;

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

        SIZE_T destinationIndex = (SIZE_T)frame * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS;
        SIZE_T ringIndex = (SIZE_T)g_syncTranslatePcmRing.readPos * SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS;
        destinationSamples[destinationIndex] = g_syncTranslatePcmRing.samples[ringIndex];
        destinationSamples[destinationIndex + 1] = g_syncTranslatePcmRing.samples[ringIndex + 1];
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
SyncTranslatePcmRingWriteSampleFromPcm16(
    _Out_writes_bytes_(sampleBytes) BYTE* dst,
    _In_ ULONG sampleBytes,
    _In_ SHORT value,
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
        *(FLOAT*)dst = (FLOAT)value / 32768.0f;
        return;
    }

    if (sampleBytes == sizeof(SHORT))
    {
        *(SHORT*)dst = value;
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
        SHORT stereoFrame[SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS] = { 0, 0 };
        (void)SyncTranslatePcmRingReadPcm16Stereo(
            (BYTE*)stereoFrame,
            SYNCTRANSLATE_VIRTUAL_AUDIO_BYTES_PER_FRAME
        );

        BYTE* framePtr = dst + ((SIZE_T)frame * bytesPerFrame);
        for (ULONG channel = 0; channel < channels; ++channel)
        {
            SHORT sample = stereoFrame[min(channel, SYNCTRANSLATE_VIRTUAL_AUDIO_CHANNELS - 1u)];
            SyncTranslatePcmRingWriteSampleFromPcm16(
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
