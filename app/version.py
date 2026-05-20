from __future__ import annotations

from importlib import metadata


PACKAGE_NAME = "synctranslate"
APP_VERSION = "2.1.0"


def app_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return APP_VERSION


__all__ = ["APP_VERSION", "PACKAGE_NAME", "app_version"]
