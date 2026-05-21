<#
.SYNOPSIS
    Creates a root-enumerated PnP device node for SyncTranslate Virtual Audio.
    Used as a devcon-free alternative inside the MSI custom action.
.PARAMETER HardwareId
    The hardware ID to register, e.g. "Root\SyncTranslateVirtualAudio".
.PARAMETER InfPath
    Path to the driver INF file to stage before creating the device node.
#>
param(
    [string]$HardwareId = "Root\SyncTranslateVirtualAudio",
    [string]$InfPath = ""
)

$ErrorActionPreference = "Continue"
$LogFile = "$env:TEMP\synctranslate_device_install.log"

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "[$timestamp] $Message"
    Write-Host $msg
    Add-Content -Path $LogFile -Value $msg -Encoding UTF8 -ErrorAction SilentlyContinue
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class RootDeviceInstaller {
    [DllImport("setupapi.dll", SetLastError = true, CharSet = CharSet.Auto)]
    static extern IntPtr SetupDiCreateDeviceInfoList(ref Guid ClassGuid, IntPtr hwndParent);

    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiDestroyDeviceInfoList(IntPtr DeviceInfoSet);

    [DllImport("setupapi.dll", SetLastError = true, CharSet = CharSet.Auto)]
    static extern bool SetupDiCreateDeviceInfo(
        IntPtr DeviceInfoSet, string DeviceName, ref Guid ClassGuid,
        string DeviceDescription, IntPtr hwndParent, int CreationFlags,
        ref SP_DEVINFO_DATA DeviceInfoData);

    [DllImport("setupapi.dll", SetLastError = true, CharSet = CharSet.Auto)]
    static extern bool SetupDiSetDeviceRegistryProperty(
        IntPtr DeviceInfoSet, ref SP_DEVINFO_DATA DeviceInfoData,
        int Property, byte[] PropertyBuffer, int PropertyBufferSize);

    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiCallClassInstaller(
        int InstallFunction, IntPtr DeviceInfoSet, ref SP_DEVINFO_DATA DeviceInfoData);

    [StructLayout(LayoutKind.Sequential)]
    public struct SP_DEVINFO_DATA {
        public int    cbSize;
        public Guid   ClassGuid;
        public int    DevInst;
        public IntPtr Reserved;
    }

    // {4d36e96c-e325-11ce-bfc1-08002be10318}  MEDIA class
    static readonly Guid GUID_DEVCLASS_MEDIA = new Guid("4d36e96c-e325-11ce-bfc1-08002be10318");
    const int SPDRP_HARDWAREID   = 0x00000001;
    const int DIF_REGISTERDEVICE = 0x00000019;
    const int DICD_GENERATE_ID   = 0x00000001;
    static readonly IntPtr INVALID_HANDLE = new IntPtr(-1);

    public static int CreateRootDevice(string hardwareId) {
        try {
            Guid classGuid = GUID_DEVCLASS_MEDIA;
            IntPtr hDevInfo = SetupDiCreateDeviceInfoList(ref classGuid, IntPtr.Zero);
            if (hDevInfo == INVALID_HANDLE) {
                int err = Marshal.GetLastWin32Error();
                return err;
            }
            try {
                SP_DEVINFO_DATA devData = new SP_DEVINFO_DATA();
                devData.cbSize = Marshal.SizeOf(devData);

                if (!SetupDiCreateDeviceInfo(
                        hDevInfo, "MEDIA", ref classGuid, null,
                        IntPtr.Zero, DICD_GENERATE_ID, ref devData)) {
                    int err = Marshal.GetLastWin32Error();
                    return err;
                }

                byte[] hwIdBytes = Encoding.Unicode.GetBytes(hardwareId + "\0\0");
                if (!SetupDiSetDeviceRegistryProperty(
                        hDevInfo, ref devData, SPDRP_HARDWAREID, hwIdBytes, hwIdBytes.Length)) {
                    int err = Marshal.GetLastWin32Error();
                    return err;
                }

                if (!SetupDiCallClassInstaller(DIF_REGISTERDEVICE, hDevInfo, ref devData)) {
                    int err = Marshal.GetLastWin32Error();
                    return err;
                }
                return 0;
            } finally {
                SetupDiDestroyDeviceInfoList(hDevInfo);
            }
        } catch {
            return -1;
        }
    }
}
'@ -Language CSharp -IgnoreWarnings

Log "[create-root-device] Starting device creation script"
Log "[create-root-device] HardwareId: $HardwareId"
Log "[create-root-device] InfPath: $InfPath"

# Stage the driver first so PnP manager can match when the device node appears
if ($InfPath -and (Test-Path $InfPath)) {
    Log "[create-root-device] staging driver: $InfPath"
    $out = & pnputil.exe /add-driver $InfPath /install 2>&1
    $out | ForEach-Object { Log "[pnputil] $_" }
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 259 -and $LASTEXITCODE -ne 5) {
        Log "[create-root-device] WARNING: pnputil returned $LASTEXITCODE, continuing anyway"
    }
} else {
    Log "[create-root-device] WARNING: INF not found or not specified: '$InfPath'. Device may not bind to a driver."
}

Log "[create-root-device] creating root device node: $HardwareId"
$result = [RootDeviceInstaller]::CreateRootDevice($HardwareId)
if ($result -eq 0) {
    Log "[create-root-device] device created successfully: $HardwareId"
    
    # Scan devices to force Windows to detect and bind the new device
    Log "[create-root-device] scanning for new devices..."
    & pnputil.exe /scan-devices 2>&1 | ForEach-Object { Log "[pnputil] $_" }
    
    Log "[create-root-device] done"
    exit 0
} else {
    Log "[create-root-device] ERROR: SetupAPI failed with error code $result"
    Log "[create-root-device] Attempting fallback: running pnputil scan-devices anyway"
    & pnputil.exe /scan-devices 2>&1 | ForEach-Object { Log "[pnputil] $_" }
    exit 1
}
