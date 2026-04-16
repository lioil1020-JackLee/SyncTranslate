# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


datas = collect_data_files("opencc") + collect_data_files("PySide6") + collect_data_files("sounddevice") + collect_data_files("soundcard")
icon = None
binaries = []
hidden_imports = []

for pkg in ("funasr", "faster_whisper", "modelscope", "addict", "torch", "torchaudio"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hidden_imports += pkg_hiddenimports


def collect_openssl_binaries():
    dll_dir = os.path.join(sys.base_prefix, "DLLs")
    names = ("libssl-3-x64.dll", "libcrypto-3-x64.dll")
    found = []
    for name in names:
        candidate = os.path.join(dll_dir, name)
        if os.path.exists(candidate):
            found.append((candidate, "."))
    return found


binaries.extend(collect_openssl_binaries())

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
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "edge_tts",
        "miniaudio",
        "opencc",
        "soundcard",
        "sounddevice",
        "tiktoken",
        "addict",
        "modelscope",
        "onnxruntime",
        "torch",
        "torchaudio",
        # app sub-packages (Phase 1-4 new modules)
        "app.application.transcript_postprocessor",
        "app.application.asr_event_processor",
        "app.application.translation_dispatcher",
        "app.application.tts_dispatcher",
        "app.application.pipeline_metrics",
        "app.domain.glossary",
        "app.domain.metrics",
        "app.infra.config.glossary_loader",
        "app.infra.logging",
        "app.infra.logging.runtime_logger",
        "app.infra.asr.streaming_policy",
        "app.infra.asr.endpoint_profiles",
        "app.infra.asr.audio_pipeline",
        "app.infra.asr.audio_pipeline.base",
        "app.infra.asr.audio_pipeline.identity",
        "app.infra.asr.audio_pipeline.highpass",
        "app.infra.asr.audio_pipeline.loudness",
        "app.infra.asr.audio_pipeline.noise_reduction",
        "app.infra.asr.audio_pipeline.music_suppression",
        "app.infra.asr.audio_pipeline.frontend_chain",
        "app.ui.controllers",
        "app.ui.controllers.session_action_controller",
        "app.ui.controllers.live_caption_refresh_controller",
        "app.ui.controllers.config_hot_apply_controller",
        "app.ui.controllers.healthcheck_controller",
    ] + hidden_imports + collect_submodules("funasr") + collect_submodules("modelscope") + collect_submodules("faster_whisper"),
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
