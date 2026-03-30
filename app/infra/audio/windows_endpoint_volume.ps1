param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("render", "capture")]
    [string]$Flow,
    [Parameter(Mandatory = $true)]
    [string]$DeviceName,
    [Parameter(Mandatory = $true)]
    [int]$VolumePercent
)

$ErrorActionPreference = "Stop"

Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SyncTranslate.Audio {
    [ComImport]
    [Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    internal class MMDeviceEnumeratorComObject {
    }

    internal enum EDataFlow {
        eRender = 0,
        eCapture = 1,
        eAll = 2
    }

    [Flags]
    internal enum DeviceState : uint {
        Active = 0x1
    }

    [Flags]
    internal enum ClsCtx : uint {
        InprocServer = 0x1,
        InprocHandler = 0x2,
        LocalServer = 0x4,
        All = InprocServer | InprocHandler | LocalServer
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct PropertyKey {
        public Guid fmtid;
        public int pid;

        public PropertyKey(Guid formatId, int propertyId) {
            fmtid = formatId;
            pid = propertyId;
        }
    }

    [StructLayout(LayoutKind.Explicit)]
    internal struct PropVariant {
        [FieldOffset(0)]
        public ushort vt;
        [FieldOffset(8)]
        public IntPtr pointerValue;

        public string GetString() {
            return vt == 31 && pointerValue != IntPtr.Zero
                ? Marshal.PtrToStringUni(pointerValue)
                : string.Empty;
        }
    }

    internal static class NativeMethods {
        [DllImport("ole32.dll")]
        internal static extern int PropVariantClear(ref PropVariant pvar);
    }

    [ComImport]
    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IMMDeviceEnumerator {
        int EnumAudioEndpoints(EDataFlow dataFlow, DeviceState stateMask, out IMMDeviceCollection devices);
    }

    [ComImport]
    [Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IMMDeviceCollection {
        int GetCount(out uint count);
        int Item(uint index, out IMMDevice device);
    }

    [ComImport]
    [Guid("D666063F-1587-4E43-81F1-B948E807363F")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IMMDevice {
        int Activate(ref Guid iid, ClsCtx clsCtx, IntPtr activationParams, [MarshalAs(UnmanagedType.Interface)] out object endpointVolume);
        int OpenPropertyStore(int accessMode, out IPropertyStore properties);
        int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
        int GetState(out DeviceState state);
    }

    [ComImport]
    [Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IPropertyStore {
        int GetCount(out uint propertyCount);
        int GetAt(uint propertyIndex, out PropertyKey key);
        int GetValue(ref PropertyKey key, out PropVariant value);
        int SetValue(ref PropertyKey key, ref PropVariant value);
        int Commit();
    }

    [ComImport]
    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IAudioEndpointVolume {
        int RegisterControlChangeNotify(IntPtr notify);
        int UnregisterControlChangeNotify(IntPtr notify);
        int GetChannelCount(out uint channelCount);
        int SetMasterVolumeLevel(float levelDb, Guid eventContext);
        int SetMasterVolumeLevelScalar(float level, Guid eventContext);
        int GetMasterVolumeLevel(out float levelDb);
        int GetMasterVolumeLevelScalar(out float level);
        int SetChannelVolumeLevel(uint channelNumber, float levelDb, Guid eventContext);
        int SetChannelVolumeLevelScalar(uint channelNumber, float level, Guid eventContext);
        int GetChannelVolumeLevel(uint channelNumber, out float levelDb);
        int GetChannelVolumeLevelScalar(uint channelNumber, out float level);
        int SetMute([MarshalAs(UnmanagedType.Bool)] bool isMuted, Guid eventContext);
        int GetMute(out bool isMuted);
        int GetVolumeStepInfo(out uint step, out uint stepCount);
        int VolumeStepUp(Guid eventContext);
        int VolumeStepDown(Guid eventContext);
        int QueryHardwareSupport(out uint hardwareSupportMask);
        int GetVolumeRange(out float minLevelDb, out float maxLevelDb, out float incrementDb);
    }

    public static class AudioEndpointVolumeController {
        private static readonly PropertyKey FriendlyNameKey = new PropertyKey(
            new Guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"),
            14
        );
        private static readonly Guid EndpointVolumeGuid = new Guid("5CDF2C82-841E-4546-9722-0CF74078229A");

        public static string SetEndpointVolume(string flowName, string requestedDeviceName, float scalar) {
            IMMDeviceEnumerator enumerator = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
            EDataFlow flow = flowName.Equals("capture", StringComparison.OrdinalIgnoreCase)
                ? EDataFlow.eCapture
                : EDataFlow.eRender;
            IMMDeviceCollection devices;
            Marshal.ThrowExceptionForHR(enumerator.EnumAudioEndpoints(flow, DeviceState.Active, out devices));
            uint count;
            Marshal.ThrowExceptionForHR(devices.GetCount(out count));

            IMMDevice bestMatch = null;
            string bestName = string.Empty;
            string normalizedRequested = Normalize(requestedDeviceName);

            for (uint i = 0; i < count; i++) {
                IMMDevice device;
                Marshal.ThrowExceptionForHR(devices.Item(i, out device));
                string friendlyName = GetFriendlyName(device);
                if (friendlyName.Equals(requestedDeviceName, StringComparison.OrdinalIgnoreCase)) {
                    bestMatch = device;
                    bestName = friendlyName;
                    break;
                }
                if (bestMatch == null && Normalize(friendlyName) == normalizedRequested) {
                    bestMatch = device;
                    bestName = friendlyName;
                }
            }

            if (bestMatch == null) {
                throw new InvalidOperationException("Audio endpoint not found: " + requestedDeviceName);
            }

            object endpointObject;
            Guid endpointVolumeGuid = EndpointVolumeGuid;
            Marshal.ThrowExceptionForHR(bestMatch.Activate(ref endpointVolumeGuid, ClsCtx.All, IntPtr.Zero, out endpointObject));
            IAudioEndpointVolume endpoint = (IAudioEndpointVolume)endpointObject;
            float bounded = Math.Max(0.0f, Math.Min(1.0f, scalar));
            Marshal.ThrowExceptionForHR(endpoint.SetMasterVolumeLevelScalar(bounded, Guid.Empty));
            return bestName;
        }

        private static string GetFriendlyName(IMMDevice device) {
            IPropertyStore properties;
            Marshal.ThrowExceptionForHR(device.OpenPropertyStore(0, out properties));
            PropVariant value = new PropVariant();
            PropertyKey friendlyNameKey = FriendlyNameKey;
            try {
                Marshal.ThrowExceptionForHR(properties.GetValue(ref friendlyNameKey, out value));
                return value.GetString() ?? string.Empty;
            }
            finally {
                NativeMethods.PropVariantClear(ref value);
            }
        }

        private static string Normalize(string value) {
            if (string.IsNullOrWhiteSpace(value)) {
                return string.Empty;
            }
            var builder = new StringBuilder();
            foreach (var ch in value.Trim().ToLowerInvariant()) {
                if (char.IsLetterOrDigit(ch)) {
                    builder.Append(ch);
                }
            }
            return builder.ToString();
        }
    }
}
"@

$scalar = [Math]::Max(0.0, [Math]::Min(1.0, $VolumePercent / 100.0))
[SyncTranslate.Audio.AudioEndpointVolumeController]::SetEndpointVolume($Flow, $DeviceName, [float]$scalar) | Out-Null
