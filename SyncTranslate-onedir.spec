# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("opencc")
icon = None

here = os.path.abspath(".")
lioil_ico = os.path.join(here, "lioil.ico")
lioil_icns = os.path.join(here, "lioil.icns")
config_yaml = os.path.join(here, "config.yaml")

if os.path.exists(lioil_ico):
    datas.append((lioil_ico, "."))
    icon = lioil_ico
elif os.path.exists(lioil_icns):
    datas.append((lioil_icns, "."))
    icon = lioil_icns

if os.path.exists(config_yaml):
    datas.append((config_yaml, "."))

endpoint_volume_script = os.path.join(here, "app", "infra", "audio", "windows_endpoint_volume.ps1")
if os.path.exists(endpoint_volume_script):
    datas.append((endpoint_volume_script, os.path.join("app", "infra", "audio")))

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "edge_tts",
        "faster_whisper",
        "miniaudio",
        "opencc",
        "soundcard",
        "sounddevice",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SyncTranslate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SyncTranslate-onedir",
)
