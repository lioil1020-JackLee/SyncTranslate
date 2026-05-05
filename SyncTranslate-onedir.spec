# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files


datas = collect_data_files("opencc") + collect_data_files("PySide6") + collect_data_files("sounddevice") + collect_data_files("soundcard")
icon = None
binaries = []
hidden_imports = []


def collect_openssl_binaries():
    dll_dir = os.path.join(sys.base_prefix, "DLLs")
    names = ("libssl-3-x64.dll", "libcrypto-3-x64.dll")
    found = []
    for name in names:
        candidate = os.path.join(dll_dir, name)
        if os.path.exists(candidate):
            found.append((candidate, "."))
    return found


# NOTE: External runtimes (runtimes/shared, runtimes/faster_whisper)
# and model files under runtimes/models are copied by relocate_ai_runtime_artifacts.ps1.
# This keeps AI packages isolated from .venv and makes rebuilds faster - they're copied
# directly from the pre-built runtimes/ directory in the dev environment.

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

for runtime_script in (
    "prepare_external_runtimes.ps1",
    "relocate_ai_runtime_artifacts.ps1",
    "download_llm_model.py",
    "download_belle_model.py",
    "verify_llama_runtime.py",
    "package_onedir.ps1",
):
    script_path = os.path.join(here, "tools", "runtime_setup", runtime_script)
    if os.path.exists(script_path):
        datas.append((script_path, os.path.join("tools", "runtime_setup")))

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
        "pickletools",
        "sqlite3",
        "soundcard",
        "sounddevice",
        # app sub-packages
        "app.application.transcript_postprocessor",
        "app.application.translation_dispatcher",
        "app.domain.glossary",
        "app.infra.config.glossary_loader",
        "app.infra.config.schema",
        "app.infra.config.settings_store",
        "app.infra.config._config_migration",
        "app.infra.config._config_serialization",
        "app.infra.config._schema_parser",
        "app.infra.translation.engine",
        "app.infra.translation.inprocess_adapter",
        "app.infra.translation.provider",
        "app.infra.translation.stitcher",
        "app.infra.translation._prompt_builder",
        "app.infra.translation._stream_parser",
        "app.infra.tts.edge_tts_adapter",
        "app.infra.tts.playback_queue",
        "app.infra.tts.voice_policy",
        "app.infra.subprocess_utils",
        "app.infra.asr.streaming_policy",
        "app.infra.asr.endpoint_profiles",
        "app.infra.asr.backend_v2",
        "app.infra.asr.backend_resolution",
        "app.infra.asr.contracts",
        "app.infra.asr.endpointing_v2",
        "app.infra.asr.enhancement_v2",
        "app.infra.asr.faster_whisper_adapter",
        "app.infra.asr.frontend_v2",
        "app.infra.asr.language_profiles",
        "app.infra.asr.lexical_bias_v2",
        "app.infra.asr.manager_v2",
        "app.infra.asr.pipeline_v2",
        "app.infra.asr.profile_selection",
        "app.infra.asr.resampling",
        "app.infra.asr.speaker_diarizer",
        "app.infra.asr.text_correction",
        "app.infra.asr.transcript_validator_v2",
        "app.infra.asr.worker_v2",
        "app.infra.asr._adaptive_tuner",
        "app.infra.asr._hallucination_filter",
        "app.domain.constants",
        "app.domain.models",
        "app.domain.runtime_state",
        "app.domain.unicode_utils",
        "app.application.audio_router",
        "app.application.config_apply_service",
        "app.application.healthcheck_service",
        "app.application.session_service",
        "app.application.settings_service",
        "app.application.transcript_service",
    ] + hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "faster_whisper",
        "modelscope",
        "onnxruntime",
        "ctranslate2",
        "tiktoken",
    ],
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
