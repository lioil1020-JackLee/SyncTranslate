from __future__ import annotations

import ctypes
from ctypes import wintypes


WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102


class WindowsNamedEvent:
    def __init__(self, name: str, *, manual_reset: bool = True, initial_state: bool = False) -> None:
        self.name = str(name)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._create_event = kernel32.CreateEventW
        self._create_event.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        self._create_event.restype = wintypes.HANDLE
        self._set_event = kernel32.SetEvent
        self._set_event.argtypes = [wintypes.HANDLE]
        self._set_event.restype = wintypes.BOOL
        self._reset_event = kernel32.ResetEvent
        self._reset_event.argtypes = [wintypes.HANDLE]
        self._reset_event.restype = wintypes.BOOL
        self._wait_for_single_object = kernel32.WaitForSingleObject
        self._wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._wait_for_single_object.restype = wintypes.DWORD
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL

        handle = self._create_event(None, bool(manual_reset), bool(initial_state), self.name)
        if not handle:
            error = ctypes.get_last_error()
            raise OSError(error, f"CreateEventW failed for {self.name}")
        self._handle = handle

    def set(self) -> None:
        if not self._set_event(self._handle):
            error = ctypes.get_last_error()
            raise OSError(error, f"SetEvent failed for {self.name}")

    def reset(self) -> None:
        if not self._reset_event(self._handle):
            error = ctypes.get_last_error()
            raise OSError(error, f"ResetEvent failed for {self.name}")

    def wait(self, timeout_ms: int = 0) -> bool:
        result = self._wait_for_single_object(self._handle, int(timeout_ms))
        if result == WAIT_OBJECT_0:
            return True
        if result == WAIT_TIMEOUT:
            return False
        raise OSError(ctypes.get_last_error(), f"WaitForSingleObject failed for {self.name}")

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._close_handle(handle)
            self._handle = None

    def __enter__(self) -> "WindowsNamedEvent":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


__all__ = ["WindowsNamedEvent"]
