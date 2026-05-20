from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np


class SyncTranslateDriverUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SyncTranslateDriverStats:
    capacity_frames: int
    buffered_frames: int
    total_written_frames: int
    total_read_frames: int
    dropped_frames: int
    underrun_frames: int


class _KernelStats(ctypes.Structure):
    _fields_ = [
        ("capacityFrames", ctypes.c_ulonglong),
        ("bufferedFrames", ctypes.c_ulonglong),
        ("totalWrittenFrames", ctypes.c_ulonglong),
        ("totalReadFrames", ctypes.c_ulonglong),
        ("droppedFrames", ctypes.c_ulonglong),
        ("underrunFrames", ctypes.c_ulonglong),
    ]


class SyncTranslateDriverAudioClient:
    DEVICE_PATH = r"\\.\SyncTranslateVirtualAudioControl"
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_DEVICE_UNKNOWN = 0x00000022
    METHOD_BUFFERED = 0
    FILE_READ_DATA = 0x0001
    FILE_WRITE_DATA = 0x0002

    IOCTL_WRITE_PCM = (
        (FILE_DEVICE_UNKNOWN << 16)
        | (FILE_WRITE_DATA << 14)
        | (0x801 << 2)
        | METHOD_BUFFERED
    )
    IOCTL_FLUSH = (
        (FILE_DEVICE_UNKNOWN << 16)
        | (FILE_WRITE_DATA << 14)
        | (0x802 << 2)
        | METHOD_BUFFERED
    )
    IOCTL_GET_STATS = (
        (FILE_DEVICE_UNKNOWN << 16)
        | (FILE_READ_DATA << 14)
        | (0x803 << 2)
        | METHOD_BUFFERED
    )

    def __init__(self, *, device_path: str = DEVICE_PATH) -> None:
        self.device_path = str(device_path)
        self._handle: int | None = None
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
        if self._kernel32 is not None:
            self._kernel32.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            self._kernel32.CreateFileW.restype = wintypes.HANDLE
            self._kernel32.DeviceIoControl.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPVOID,
            ]
            self._kernel32.DeviceIoControl.restype = wintypes.BOOL
            self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._kernel32.CloseHandle.restype = wintypes.BOOL

    def is_available(self) -> bool:
        try:
            self._ensure_handle()
            return True
        except SyncTranslateDriverUnavailable:
            return False

    def write_virtual_microphone(self, audio: np.ndarray, *, sample_rate: int) -> int:
        del sample_rate
        payload = self._to_float32_mono(audio)
        if payload.size == 0:
            return 0
        written = self._device_io_control(self.IOCTL_WRITE_PCM, payload, None)
        return int(written // np.dtype(np.float32).itemsize)

    def flush_virtual_microphone(self) -> None:
        self._device_io_control(self.IOCTL_FLUSH, None, None)

    def stats(self) -> SyncTranslateDriverStats:
        raw = _KernelStats()
        self._device_io_control(self.IOCTL_GET_STATS, None, raw)
        return SyncTranslateDriverStats(
            capacity_frames=int(raw.capacityFrames),
            buffered_frames=int(raw.bufferedFrames),
            total_written_frames=int(raw.totalWrittenFrames),
            total_read_frames=int(raw.totalReadFrames),
            dropped_frames=int(raw.droppedFrames),
            underrun_frames=int(raw.underrunFrames),
        )

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None and self._kernel32 is not None:
            self._kernel32.CloseHandle(wintypes.HANDLE(handle))

    def _ensure_handle(self) -> int:
        if self._handle is not None:
            return self._handle
        if self._kernel32 is None:
            raise SyncTranslateDriverUnavailable("synctranslate_driver_control_requires_windows")

        handle = self._kernel32.CreateFileW(
            self.device_path,
            self.GENERIC_READ | self.GENERIC_WRITE,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            None,
            self.OPEN_EXISTING,
            0,
            None,
        )
        handle_value = int(handle)
        if handle_value == int(self.INVALID_HANDLE_VALUE):
            error = ctypes.get_last_error()
            raise SyncTranslateDriverUnavailable(f"synctranslate_driver_control_unavailable:{error}")
        self._handle = handle_value
        return handle_value

    def _device_io_control(self, control_code: int, input_data: np.ndarray | None, output_struct) -> int:
        handle = self._ensure_handle()
        input_ptr = None
        input_bytes = 0
        if input_data is not None:
            contiguous = np.ascontiguousarray(input_data, dtype=np.float32)
            input_ptr = contiguous.ctypes.data_as(wintypes.LPVOID)
            input_bytes = int(contiguous.nbytes)
        output_ptr = None
        output_bytes = 0
        if output_struct is not None:
            output_ptr = ctypes.byref(output_struct)
            output_bytes = ctypes.sizeof(output_struct)

        bytes_returned = wintypes.DWORD(0)
        ok = self._kernel32.DeviceIoControl(
            wintypes.HANDLE(handle),
            wintypes.DWORD(control_code),
            input_ptr,
            wintypes.DWORD(input_bytes),
            output_ptr,
            wintypes.DWORD(output_bytes),
            ctypes.byref(bytes_returned),
            None,
        )
        if not ok:
            error = ctypes.get_last_error()
            if self._handle is not None:
                self.close()
            raise SyncTranslateDriverUnavailable(f"synctranslate_driver_ioctl_failed:{error}")
        return int(bytes_returned.value)

    @staticmethod
    def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
        payload = np.asarray(audio, dtype=np.float32)
        if payload.size == 0:
            return np.empty((0,), dtype=np.float32)
        if payload.ndim == 1:
            mono = payload
        else:
            mono = np.mean(payload, axis=1, dtype=np.float32)
        mono = np.ascontiguousarray(mono.reshape(-1), dtype=np.float32)
        np.clip(mono, -1.0, 1.0, out=mono)
        return mono


__all__ = [
    "SyncTranslateDriverAudioClient",
    "SyncTranslateDriverStats",
    "SyncTranslateDriverUnavailable",
]
