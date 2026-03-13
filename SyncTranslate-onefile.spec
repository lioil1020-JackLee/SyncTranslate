# -*- mode: python ; coding: utf-8 -*-
import os


datas = []
icon = None

if os.path.exists("config.example.yaml"):
    datas.append(("config.example.yaml", "."))
if os.path.exists("config.yaml"):
    datas.append(("config.yaml", "."))

if os.name == "nt" and os.path.exists("lioil.ico"):
    datas.append(("lioil.ico", "."))
    icon = "lioil.ico"
elif os.path.exists("lioil.icns"):
    datas.append(("lioil.icns", "."))
    icon = "lioil.icns"

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
        "soundcard",
        "sounddevice",
        "pycaw.pycaw",
        "comtypes",
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
    a.binaries,
    a.datas,
    [],
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
