from __future__ import annotations

import os
import sys


def get_env_var(name: str, default: str = "") -> str:
    key = (name or "").strip()
    if not key:
        return default

    value = os.getenv(key, "").strip()
    if value:
        return value

    value = _get_windows_registry_env_value(key)
    if value:
        return value
    return default


def list_env_var_names() -> list[str]:
    names: set[str] = set(os.environ.keys())
    names.update(_list_windows_registry_env_names())
    return sorted(names, key=str.lower)


def _get_windows_registry_env_value(name: str) -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        import winreg  # type: ignore
    except Exception:
        return ""

    locations = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    access_flags = [winreg.KEY_READ]
    if hasattr(winreg, "KEY_WOW64_64KEY"):
        access_flags.append(winreg.KEY_READ | winreg.KEY_WOW64_64KEY)

    for hive, path in locations:
        for access in access_flags:
            try:
                with winreg.OpenKey(hive, path, 0, access) as key:
                    value, _ = winreg.QueryValueEx(key, name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except OSError:
                continue
    return ""


def _list_windows_registry_env_names() -> set[str]:
    if not sys.platform.startswith("win"):
        return set()
    try:
        import winreg  # type: ignore
    except Exception:
        return set()

    names: set[str] = set()
    locations = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    access_flags = [winreg.KEY_READ]
    if hasattr(winreg, "KEY_WOW64_64KEY"):
        access_flags.append(winreg.KEY_READ | winreg.KEY_WOW64_64KEY)

    for hive, path in locations:
        for access in access_flags:
            try:
                with winreg.OpenKey(hive, path, 0, access) as key:
                    index = 0
                    while True:
                        try:
                            item_name, _, _ = winreg.EnumValue(key, index)
                        except OSError:
                            break
                        if item_name:
                            names.add(item_name)
                        index += 1
            except OSError:
                continue
    return names
