from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import struct
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if os.name == "nt":
    from ctypes import wintypes
else:  # pragma: no cover - Windows-only helper.
    wintypes = None  # type: ignore[assignment]


EXPECTED_SAMPLE_RATE = 48000
EXPECTED_CHANNELS = 2
EXPECTED_BITS = 16
EXPECTED_DTYPE = "PCM16"
EXPECTED_SUMMARY = "48000Hz PCM16 2ch"


@dataclass(frozen=True, slots=True)
class WasapiEndpointFormat:
    flow: str
    name: str
    sample_rate: int
    channels: int
    bits_per_sample: int
    valid_bits_per_sample: int
    format_tag: int
    subformat: str
    dtype: str
    source: str = "device_format"

    @property
    def matches_v2(self) -> bool:
        bits = int(self.valid_bits_per_sample or self.bits_per_sample)
        return (
            int(self.sample_rate) == EXPECTED_SAMPLE_RATE
            and int(self.channels) == EXPECTED_CHANNELS
            and bits == EXPECTED_BITS
            and self.dtype == EXPECTED_DTYPE
        )


@dataclass(frozen=True, slots=True)
class WasapiFormatReport:
    status: str
    message: str
    expected: dict[str, Any]
    endpoints: list[WasapiEndpointFormat]
    error: str = ""
    shared_mix_formats: list[WasapiEndpointFormat] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "expected": dict(self.expected),
            "error": self.error,
            "endpoints": [asdict(endpoint) | {"matches_v2": endpoint.matches_v2} for endpoint in self.endpoints],
            "shared_mix_formats": [
                asdict(endpoint) | {"matches_v2": endpoint.matches_v2} for endpoint in self.shared_mix_formats
            ],
        }


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)

    def __str__(self) -> str:
        return str(uuid.UUID(bytes_le=bytes(self)))


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", ctypes.c_ulong)]


class BLOB(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("pBlobData", ctypes.c_void_p),
    ]


class _PROPVARIANT_VALUE(ctypes.Union):
    _fields_ = [
        ("pwszVal", ctypes.c_wchar_p),
        ("pszVal", ctypes.c_char_p),
        ("ulVal", ctypes.c_ulong),
        ("boolVal", ctypes.c_short),
        ("blob", BLOB),
    ]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("value", _PROPVARIANT_VALUE),
    ]


class WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", ctypes.c_ushort),
        ("nChannels", ctypes.c_ushort),
        ("nSamplesPerSec", ctypes.c_ulong),
        ("nAvgBytesPerSec", ctypes.c_ulong),
        ("nBlockAlign", ctypes.c_ushort),
        ("wBitsPerSample", ctypes.c_ushort),
        ("cbSize", ctypes.c_ushort),
    ]


class WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Format", WAVEFORMATEX),
        ("wValidBitsPerSample", ctypes.c_ushort),
        ("dwChannelMask", ctypes.c_ulong),
        ("SubFormat", GUID),
    ]


COMFUNC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
HRESULT = ctypes.c_long
ULONG = ctypes.c_ulong
DWORD = ctypes.c_ulong

CLSID_MMDEVICE_ENUMERATOR = GUID.from_string("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
IID_IMMDEVICE_ENUMERATOR = GUID.from_string("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
IID_IAUDIO_CLIENT = GUID.from_string("{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}")
PKEY_DEVICE_FRIENDLY_NAME = PROPERTYKEY(
    GUID.from_string("{A45C254E-DF1C-4EFD-8020-67D146A850E0}"),
    14,
)
PKEY_AUDIOENGINE_DEVICE_FORMAT = PROPERTYKEY(
    GUID.from_string("{F19F064D-082C-4E27-BC73-6882A1BB8E4C}"),
    0,
)
KSDATAFORMAT_SUBTYPE_PCM = GUID.from_string("{00000001-0000-0010-8000-00AA00389B71}")
KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = GUID.from_string("{00000003-0000-0010-8000-00AA00389B71}")

CLSCTX_ALL = 0x17
DEVICE_STATE_ACTIVE = 0x1
STGM_READ = 0
VT_LPWSTR = 31
VT_BLOB = 65
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def query_synctranslate_endpoint_formats() -> WasapiFormatReport:
    expected = {
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "channels": EXPECTED_CHANNELS,
        "bit_depth": EXPECTED_BITS,
        "dtype": EXPECTED_DTYPE,
        "summary": EXPECTED_SUMMARY,
    }
    if platform.system() != "Windows" or wintypes is None:
        return WasapiFormatReport("WARN", "WASAPI endpoint format probe is only available on Windows.", expected, [])
    try:
        endpoints, shared_mix_formats = _query_formats()
    except Exception as exc:
        return WasapiFormatReport("WARN", f"WASAPI endpoint format probe failed: {exc}", expected, [], str(exc))

    render = [endpoint for endpoint in endpoints if endpoint.flow == "render"]
    capture = [endpoint for endpoint in endpoints if endpoint.flow == "capture"]
    if not render or not capture:
        return WasapiFormatReport(
            "WARN",
            "SyncTranslate render/capture endpoint device formats were not found.",
            expected,
            endpoints,
            shared_mix_formats=shared_mix_formats,
        )
    mismatches = [endpoint for endpoint in endpoints if not endpoint.matches_v2]
    if mismatches:
        names = ", ".join(f"{item.flow}:{item.name}" for item in mismatches)
        return WasapiFormatReport(
            "FAIL",
            f"Endpoint device format mismatch for {names}; expected {EXPECTED_SUMMARY}.",
            expected,
            endpoints,
            shared_mix_formats=shared_mix_formats,
        )
    return WasapiFormatReport(
        "PASS",
        f"SyncTranslate endpoint device formats match {EXPECTED_SUMMARY}.",
        expected,
        endpoints,
        shared_mix_formats=shared_mix_formats,
    )


def _query_formats() -> tuple[list[WasapiEndpointFormat], list[WasapiEndpointFormat]]:
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, DWORD]
    ole32.CoInitializeEx.restype = HRESULT
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(GUID),
        ctypes.c_void_p,
        DWORD,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = HRESULT
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None
    ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
    ole32.PropVariantClear.restype = HRESULT

    coinit = ole32.CoInitializeEx(None, 0)
    co_initialized = coinit in (0, 1)
    if coinit < 0 and coinit != -2147417850:  # RPC_E_CHANGED_MODE
        _check_hr(coinit, "CoInitializeEx")

    enumerator = ctypes.c_void_p()
    try:
        _check_hr(
            ole32.CoCreateInstance(
                ctypes.byref(CLSID_MMDEVICE_ENUMERATOR),
                None,
                CLSCTX_ALL,
                ctypes.byref(IID_IMMDEVICE_ENUMERATOR),
                ctypes.byref(enumerator),
            ),
            "CoCreateInstance(IMMDeviceEnumerator)",
        )
        result: list[WasapiEndpointFormat] = []
        shared_mix_formats: list[WasapiEndpointFormat] = []
        for flow_value, flow_name in ((0, "render"), (1, "capture")):
            device_formats, mix_formats = _enum_flow_formats(enumerator, flow_value, flow_name, ole32)
            result.extend(device_formats)
            shared_mix_formats.extend(mix_formats)
        return result, shared_mix_formats
    finally:
        _release(enumerator)
        if co_initialized:
            ole32.CoUninitialize()


def _enum_flow_formats(
    enumerator: ctypes.c_void_p, flow_value: int, flow_name: str, ole32
) -> tuple[list[WasapiEndpointFormat], list[WasapiEndpointFormat]]:
    collection = ctypes.c_void_p()
    enum_audio_endpoints = _method(
        enumerator,
        3,
        HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
        DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _check_hr(enum_audio_endpoints(enumerator, flow_value, DEVICE_STATE_ACTIVE, ctypes.byref(collection)), "EnumAudioEndpoints")
    try:
        count = DWORD()
        get_count = _method(collection, 3, HRESULT, ctypes.c_void_p, ctypes.POINTER(DWORD))
        _check_hr(get_count(collection, ctypes.byref(count)), "IMMDeviceCollection.GetCount")
        endpoints: list[WasapiEndpointFormat] = []
        shared_mix_formats: list[WasapiEndpointFormat] = []
        for index in range(int(count.value)):
            device = ctypes.c_void_p()
            item = _method(collection, 4, HRESULT, ctypes.c_void_p, DWORD, ctypes.POINTER(ctypes.c_void_p))
            _check_hr(item(collection, index, ctypes.byref(device)), "IMMDeviceCollection.Item")
            try:
                name = _friendly_name(device, ole32)
                if "synctranslate" not in name.lower():
                    continue
                shared_mix_formats.append(_mix_format(device, flow_name, name, ole32))
                endpoints.append(_device_format(device, flow_name, name, ole32))
            finally:
                _release(device)
        return endpoints, shared_mix_formats
    finally:
        _release(collection)


def _friendly_name(device: ctypes.c_void_p, ole32) -> str:
    store = ctypes.c_void_p()
    open_property_store = _method(device, 4, HRESULT, ctypes.c_void_p, DWORD, ctypes.POINTER(ctypes.c_void_p))
    _check_hr(open_property_store(device, STGM_READ, ctypes.byref(store)), "IMMDevice.OpenPropertyStore")
    try:
        prop = PROPVARIANT()
        get_value = _method(
            store,
            5,
            HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(PROPERTYKEY),
            ctypes.POINTER(PROPVARIANT),
        )
        _check_hr(get_value(store, ctypes.byref(PKEY_DEVICE_FRIENDLY_NAME), ctypes.byref(prop)), "IPropertyStore.GetValue")
        try:
            if int(prop.vt) == VT_LPWSTR and prop.value.pwszVal:
                return str(prop.value.pwszVal)
            return ""
        finally:
            ole32.PropVariantClear(ctypes.byref(prop))
    finally:
        _release(store)


def _device_format(device: ctypes.c_void_p, flow_name: str, name: str, ole32) -> WasapiEndpointFormat:
    store = ctypes.c_void_p()
    open_property_store = _method(device, 4, HRESULT, ctypes.c_void_p, DWORD, ctypes.POINTER(ctypes.c_void_p))
    _check_hr(open_property_store(device, STGM_READ, ctypes.byref(store)), "IMMDevice.OpenPropertyStore")
    try:
        prop = PROPVARIANT()
        get_value = _method(
            store,
            5,
            HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(PROPERTYKEY),
            ctypes.POINTER(PROPVARIANT),
        )
        _check_hr(
            get_value(store, ctypes.byref(PKEY_AUDIOENGINE_DEVICE_FORMAT), ctypes.byref(prop)),
            "IPropertyStore.GetValue(PKEY_AudioEngine_DeviceFormat)",
        )
        try:
            blob = _propvariant_blob_bytes(prop)
            return _format_from_bytes(blob, flow_name, name, source="device_format")
        finally:
            ole32.PropVariantClear(ctypes.byref(prop))
    finally:
        _release(store)


def _mix_format(device: ctypes.c_void_p, flow_name: str, name: str, ole32) -> WasapiEndpointFormat:
    audio_client = ctypes.c_void_p()
    activate = _method(
        device,
        3,
        HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(GUID),
        DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    _check_hr(activate(device, ctypes.byref(IID_IAUDIO_CLIENT), CLSCTX_ALL, None, ctypes.byref(audio_client)), "IMMDevice.Activate(IAudioClient)")
    format_ptr = ctypes.c_void_p()
    try:
        get_mix_format = _method(audio_client, 8, HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        _check_hr(get_mix_format(audio_client, ctypes.byref(format_ptr)), "IAudioClient.GetMixFormat")
        fmt = ctypes.cast(format_ptr, ctypes.POINTER(WAVEFORMATEX)).contents
        valid_bits = int(fmt.wBitsPerSample)
        subformat = ""
        dtype = _dtype_from_format_tag(int(fmt.wFormatTag), "")
        if int(fmt.wFormatTag) == WAVE_FORMAT_EXTENSIBLE and int(fmt.cbSize) >= 22:
            ext = ctypes.cast(format_ptr, ctypes.POINTER(WAVEFORMATEXTENSIBLE)).contents
            valid_bits = int(ext.wValidBitsPerSample or fmt.wBitsPerSample)
            subformat = str(ext.SubFormat)
            dtype = _dtype_from_format_tag(int(fmt.wFormatTag), subformat)
        return WasapiEndpointFormat(
            flow=flow_name,
            name=name,
            sample_rate=int(fmt.nSamplesPerSec),
            channels=int(fmt.nChannels),
            bits_per_sample=int(fmt.wBitsPerSample),
            valid_bits_per_sample=valid_bits,
            format_tag=int(fmt.wFormatTag),
            subformat=subformat,
            dtype=dtype,
            source="shared_mix_format",
        )
    finally:
        if format_ptr:
            ole32.CoTaskMemFree(format_ptr)
        _release(audio_client)


def _propvariant_blob_bytes(prop: PROPVARIANT) -> bytes:
    # VT_BLOB is the PROPVARIANT representation used for
    # PKEY_AudioEngine_DeviceFormat. The blob payload is a WAVEFORMATEX or
    # WAVEFORMATEXTENSIBLE byte sequence.
    if int(prop.vt) != VT_BLOB:
        raise RuntimeError(f"Expected VT_BLOB for PKEY_AudioEngine_DeviceFormat, got vt={int(prop.vt)}")
    size = int(prop.value.blob.cbSize)
    data_ptr = prop.value.blob.pBlobData
    if size <= 0 or not data_ptr:
        raise RuntimeError("PKEY_AudioEngine_DeviceFormat blob is empty")
    return bytes(ctypes.string_at(data_ptr, size))


def _format_from_bytes(data: bytes, flow_name: str, name: str, *, source: str) -> WasapiEndpointFormat:
    # Some registry dumps include a property-store wrapper before the raw
    # WAVEFORMAT bytes. COM's PROPVARIANT usually gives only the raw payload,
    # but probing a couple of aligned offsets keeps the parser tolerant.
    for offset in (0, 8, 16):
        if len(data) < offset + 18:
            continue
        (
            format_tag,
            channels,
            samples_per_sec,
            _avg_bytes_per_sec,
            _block_align,
            bits_per_sample,
            cb_size,
        ) = struct.unpack_from("<HHIIHHH", data, offset)
        if not _looks_like_waveformat(format_tag, channels, samples_per_sec, bits_per_sample, cb_size):
            continue
        valid_bits = int(bits_per_sample)
        subformat = ""
        dtype = _dtype_from_format_tag(int(format_tag), "")
        if int(format_tag) == WAVE_FORMAT_EXTENSIBLE and int(cb_size) >= 22 and len(data) >= offset + 40:
            valid_bits = struct.unpack_from("<H", data, offset + 18)[0] or int(bits_per_sample)
            subformat = str(uuid.UUID(bytes_le=data[offset + 24 : offset + 40]))
            dtype = _dtype_from_format_tag(int(format_tag), subformat)
        return WasapiEndpointFormat(
            flow=flow_name,
            name=name,
            sample_rate=int(samples_per_sec),
            channels=int(channels),
            bits_per_sample=int(bits_per_sample),
            valid_bits_per_sample=int(valid_bits),
            format_tag=int(format_tag),
            subformat=subformat,
            dtype=dtype,
            source=source,
        )
    raise RuntimeError("Unable to parse WAVEFORMAT bytes from PKEY_AudioEngine_DeviceFormat")


def _looks_like_waveformat(
    format_tag: int,
    channels: int,
    samples_per_sec: int,
    bits_per_sample: int,
    cb_size: int,
) -> bool:
    return (
        int(format_tag) in {WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_EXTENSIBLE}
        and 1 <= int(channels) <= 16
        and 8000 <= int(samples_per_sec) <= 384000
        and int(bits_per_sample) in {8, 16, 20, 24, 32, 64}
        and 0 <= int(cb_size) <= 64
    )


def _dtype_from_format_tag(format_tag: int, subformat: str) -> str:
    normalized_subformat = str(subformat or "").lower()
    if format_tag == WAVE_FORMAT_PCM or normalized_subformat == str(KSDATAFORMAT_SUBTYPE_PCM).lower():
        return "PCM16"
    if format_tag == WAVE_FORMAT_IEEE_FLOAT or normalized_subformat == str(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT).lower():
        return "FLOAT32"
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        return "EXTENSIBLE"
    return f"FORMAT_TAG_{format_tag}"


def _method(ptr: ctypes.c_void_p, index: int, restype, *argtypes):
    if not ptr:
        raise RuntimeError("COM pointer is null")
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return COMFUNC(restype, *argtypes)(vtbl[index])


def _release(ptr: ctypes.c_void_p) -> None:
    if not ptr:
        return
    release = _method(ptr, 2, ULONG, ctypes.c_void_p)
    release(ptr)


def _check_hr(hr: int, context: str) -> None:
    value = int(ctypes.c_long(hr).value)
    if value < 0:
        raise OSError(value, f"{context} failed")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect SyncTranslate WASAPI endpoint device formats.")
    parser.add_argument("--json", default="", help="Write JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    report = query_synctranslate_endpoint_formats()
    print(f"{report.status} WASAPI endpoint format probe")
    print(report.message)
    for endpoint in report.endpoints:
        print(
            f"[{'PASS' if endpoint.matches_v2 else 'FAIL'}] {endpoint.flow}: {endpoint.name} "
            f"{endpoint.sample_rate}Hz {endpoint.dtype} {endpoint.valid_bits_per_sample or endpoint.bits_per_sample}bit "
            f"{endpoint.channels}ch"
        )
    for endpoint in report.shared_mix_formats:
        print(
            f"[INFO] {endpoint.flow} shared mix: {endpoint.name} "
            f"{endpoint.sample_rate}Hz {endpoint.dtype} {endpoint.valid_bits_per_sample or endpoint.bits_per_sample}bit "
            f"{endpoint.channels}ch"
        )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON written: {path}")
    return 1 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
