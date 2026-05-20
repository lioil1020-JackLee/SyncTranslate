#include "synctranslate_control.h"
#include "synctranslate_pcm_ring.h"

static PDEVICE_OBJECT g_syncTranslateControlDevice = NULL;
static PDRIVER_DISPATCH g_syncTranslatePreviousDeviceControl = NULL;

static NTSTATUS
SyncTranslateForwardDeviceControl(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp
)
{
    if (g_syncTranslatePreviousDeviceControl != NULL)
    {
        return g_syncTranslatePreviousDeviceControl(deviceObject, irp);
    }

    irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
    irp->IoStatus.Information = 0;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return STATUS_INVALID_DEVICE_REQUEST;
}

static NTSTATUS
SyncTranslateForwardIrp(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp,
    _In_opt_ PDRIVER_DISPATCH previousDispatch
)
{
    if (previousDispatch != NULL)
    {
        return previousDispatch(deviceObject, irp);
    }

    irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
    irp->IoStatus.Information = 0;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return STATUS_INVALID_DEVICE_REQUEST;
}

extern "C"
NTSTATUS
SyncTranslateControlDispatchCreateClose(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp
)
{
    UNREFERENCED_PARAMETER(deviceObject);
    irp->IoStatus.Status = STATUS_SUCCESS;
    irp->IoStatus.Information = 0;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

extern "C"
NTSTATUS
SyncTranslateControlDispatchClose(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp
)
{
    UNREFERENCED_PARAMETER(deviceObject);
    irp->IoStatus.Status = STATUS_SUCCESS;
    irp->IoStatus.Information = 0;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

extern "C"
NTSTATUS
SyncTranslateControlDispatchCleanup(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp
)
{
    UNREFERENCED_PARAMETER(deviceObject);
    irp->IoStatus.Status = STATUS_SUCCESS;
    irp->IoStatus.Information = 0;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

extern "C"
NTSTATUS
SyncTranslateControlDispatchDeviceControl(
    _In_ PDEVICE_OBJECT deviceObject,
    _Inout_ PIRP irp
)
{
    if (deviceObject != g_syncTranslateControlDevice)
    {
        return SyncTranslateForwardDeviceControl(deviceObject, irp);
    }

    NTSTATUS status = STATUS_SUCCESS;
    ULONG_PTR information = 0;
    PIO_STACK_LOCATION stack = IoGetCurrentIrpStackLocation(irp);
    ULONG controlCode = stack->Parameters.DeviceIoControl.IoControlCode;
    PVOID systemBuffer = irp->AssociatedIrp.SystemBuffer;

    switch (controlCode)
    {
    case IOCTL_SYNCTRANSLATE_AUDIO_WRITE_PCM:
    {
        ULONG inputBytes = stack->Parameters.DeviceIoControl.InputBufferLength;
        if (systemBuffer == NULL || inputBytes == 0 || (inputBytes % sizeof(FLOAT)) != 0)
        {
            status = STATUS_INVALID_PARAMETER;
            break;
        }

        status = SyncTranslatePcmRingInitialize(SYNCTRANSLATE_PCM_RING_DEFAULT_FRAMES);
        if (!NT_SUCCESS(status) && status != STATUS_DEVICE_BUSY)
        {
            break;
        }

        ULONG frames = inputBytes / sizeof(FLOAT);
        ULONG written = SyncTranslatePcmRingWriteFloat32Mono((const FLOAT*)systemBuffer, frames);
        information = (ULONG_PTR)written * sizeof(FLOAT);
        status = STATUS_SUCCESS;
        break;
    }

    case IOCTL_SYNCTRANSLATE_AUDIO_FLUSH:
        SyncTranslatePcmRingFlush();
        status = STATUS_SUCCESS;
        break;

    case IOCTL_SYNCTRANSLATE_AUDIO_GET_STATS:
    {
        ULONG outputBytes = stack->Parameters.DeviceIoControl.OutputBufferLength;
        if (systemBuffer == NULL || outputBytes < sizeof(SYNCTRANSLATE_PCM_RING_STATS))
        {
            status = STATUS_BUFFER_TOO_SMALL;
            break;
        }

        SyncTranslatePcmRingGetStats((SYNCTRANSLATE_PCM_RING_STATS*)systemBuffer);
        information = sizeof(SYNCTRANSLATE_PCM_RING_STATS);
        status = STATUS_SUCCESS;
        break;
    }

    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        break;
    }

    irp->IoStatus.Status = status;
    irp->IoStatus.Information = information;
    IoCompleteRequest(irp, IO_NO_INCREMENT);
    return status;
}

NTSTATUS
SyncTranslateControlInitialize(
    _In_ PDRIVER_OBJECT driverObject
)
{
    PAGED_CODE();

    if (driverObject == NULL || g_syncTranslateControlDevice != NULL)
    {
        return STATUS_SUCCESS;
    }

    UNICODE_STRING deviceName;
    UNICODE_STRING symbolicLink;
    RtlInitUnicodeString(&deviceName, SYNCTRANSLATE_CONTROL_DEVICE_NAME);
    RtlInitUnicodeString(&symbolicLink, SYNCTRANSLATE_CONTROL_SYMBOLIC_LINK);

    PDEVICE_OBJECT deviceObject = NULL;
    NTSTATUS status = IoCreateDevice(
        driverObject,
        0,
        &deviceName,
        FILE_DEVICE_UNKNOWN,
        FILE_DEVICE_SECURE_OPEN,
        FALSE,
        &deviceObject
    );
    if (!NT_SUCCESS(status))
    {
        return status;
    }

    deviceObject->Flags |= DO_BUFFERED_IO;
    status = IoCreateSymbolicLink(&symbolicLink, &deviceName);
    if (!NT_SUCCESS(status))
    {
        IoDeleteDevice(deviceObject);
        return status;
    }

    g_syncTranslateControlDevice = deviceObject;
    g_syncTranslatePreviousDeviceControl = driverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL];
    driverObject->MajorFunction[IRP_MJ_CREATE] = SyncTranslateControlDispatchCreateClose;
    driverObject->MajorFunction[IRP_MJ_CLOSE] = SyncTranslateControlDispatchClose;
    driverObject->MajorFunction[IRP_MJ_CLEANUP] = SyncTranslateControlDispatchCleanup;
    driverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = SyncTranslateControlDispatchDeviceControl;
    deviceObject->Flags &= ~DO_DEVICE_INITIALIZING;
    return STATUS_SUCCESS;
}

VOID
SyncTranslateControlShutdown(
    _In_opt_ PDRIVER_OBJECT driverObject
)
{
    PAGED_CODE();

    if (driverObject != NULL &&
        driverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] == SyncTranslateControlDispatchDeviceControl)
    {
        driverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = g_syncTranslatePreviousDeviceControl;
    }
    g_syncTranslatePreviousDeviceControl = NULL;

    UNICODE_STRING symbolicLink;
    RtlInitUnicodeString(&symbolicLink, SYNCTRANSLATE_CONTROL_SYMBOLIC_LINK);
    IoDeleteSymbolicLink(&symbolicLink);

    if (g_syncTranslateControlDevice != NULL)
    {
        IoDeleteDevice(g_syncTranslateControlDevice);
        g_syncTranslateControlDevice = NULL;
    }
}
