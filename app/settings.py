from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

from app.schemas import AppConfig


EMBEDDED_DEFAULT_CONFIG_YAML = """audio:
    meeting_in: Windows WASAPI::CABLE Output (VB-Audio Virtual Cable)
    microphone_in: Windows WASAPI::耳機 (WH-1000XM5)
    speaker_out: Windows WASAPI::耳機 (WH-1000XM5)
    meeting_out: Windows WASAPI::CABLE Input (VB-Audio Virtual Cable)
    meeting_in_gain: 1.0
    microphone_in_gain: 1.0
    speaker_out_volume: 1.0
    meeting_out_volume: 1.0
direction:
    mode: bidirectional
language:
    meeting_source: en
    meeting_target: zh-TW
    local_source: zh-TW
    local_target: en
asr:
    engine: faster_whisper
    model: large-v3
    device: cuda
    compute_type: float16
    beam_size: 1
    condition_on_previous_text: true
    temperature_fallback: 0.0,0.2
    no_speech_threshold: 0.6
    vad:
        enabled: true
        min_speech_duration_ms: 160
        min_silence_duration_ms: 420
        max_speech_duration_s: 8.0
        speech_pad_ms: 320
        rms_threshold: 0.03
    streaming:
        partial_interval_ms: 200
        partial_history_seconds: 2
        final_history_seconds: 4
asr_channels:
    chinese:
        engine: faster_whisper
        model: large-v3
        device: cuda
        compute_type: float16
        beam_size: 2
        condition_on_previous_text: true
        temperature_fallback: 0.0,0.2
        no_speech_threshold: 0.65
        vad:
            enabled: true
            min_speech_duration_ms: 170
            min_silence_duration_ms: 480
            max_speech_duration_s: 8.0
            speech_pad_ms: 340
            rms_threshold: 0.03
        streaming:
            partial_interval_ms: 220
            partial_history_seconds: 2
            final_history_seconds: 4
    english:
        engine: faster_whisper
        model: distil-large-v3
        device: cuda
        compute_type: float16
        beam_size: 1
        condition_on_previous_text: true
        temperature_fallback: 0.0,0.2,0.4
        no_speech_threshold: 0.55
        vad:
            enabled: true
            min_speech_duration_ms: 150
            min_silence_duration_ms: 400
            max_speech_duration_s: 8.0
            speech_pad_ms: 300
            rms_threshold: 0.03
        streaming:
            partial_interval_ms: 180
            partial_history_seconds: 2
            final_history_seconds: 4
llm:
    backend: lm_studio
    base_url: http://127.0.0.1:1234
    model: hy-mt1.5-7b
    temperature: 0.05
    top_p: 0.9
    max_output_tokens: 128
    repeat_penalty: 1.05
    stop_tokens: '</target>,Translation:'
    request_timeout_sec: 12
    sliding_window:
        enabled: true
        trigger_tokens: 20
        max_context_items: 4
    profiles:
        live_caption_fast:
            name: live_caption_fast
            prompt_style: literal
            context_items: 4
            partial_trigger_tokens: 18
            max_tokens: 256
            preserve_terms: true
            naturalize_tone: false
            allow_subject_completion: false
        live_caption_stable:
            name: live_caption_fast
            prompt_style: literal
            context_items: 4
            partial_trigger_tokens: 18
            max_tokens: 256
            preserve_terms: true
            naturalize_tone: false
            allow_subject_completion: false
        speech_output_natural:
            name: live_caption_fast
            prompt_style: literal
            context_items: 4
            partial_trigger_tokens: 18
            max_tokens: 256
            preserve_terms: true
            naturalize_tone: false
            allow_subject_completion: false
        technical_meeting:
            name: live_caption_fast
            prompt_style: literal
            context_items: 4
            partial_trigger_tokens: 18
            max_tokens: 256
            preserve_terms: true
            naturalize_tone: false
            allow_subject_completion: false
    caption_profile: live_caption_fast
    speech_profile: speech_output_natural
llm_channels:
    zh_to_en:
        backend: lm_studio
        base_url: http://127.0.0.1:1234
        model: hy-mt1.5-7b
        temperature: 0.05
        top_p: 0.9
        max_output_tokens: 96
        repeat_penalty: 1.05
        stop_tokens: '</target>,Translation:'
        request_timeout_sec: 12
        sliding_window:
            enabled: true
            trigger_tokens: 20
            max_context_items: 4
        profiles:
            live_caption_fast:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            live_caption_stable:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            speech_output_natural:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            technical_meeting:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
        caption_profile: live_caption_fast
        speech_profile: speech_output_natural
    en_to_zh:
        backend: lm_studio
        base_url: http://127.0.0.1:1234
        model: hy-mt1.5-7b
        temperature: 0.05
        top_p: 0.9
        max_output_tokens: 128
        repeat_penalty: 1.05
        stop_tokens: '</target>,翻譯:'
        request_timeout_sec: 12
        sliding_window:
            enabled: true
            trigger_tokens: 20
            max_context_items: 4
        profiles:
            live_caption_fast:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            live_caption_stable:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            speech_output_natural:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
            technical_meeting:
                name: live_caption_fast
                prompt_style: literal
                context_items: 4
                partial_trigger_tokens: 18
                max_tokens: 256
                preserve_terms: true
                naturalize_tone: false
                allow_subject_completion: false
        caption_profile: live_caption_fast
        speech_profile: speech_output_natural
tts:
    engine: edge_tts
    executable_path: ''
    model_path: ''
    config_path: ''
    voice_name: zh-TW-HsiaoChenNeural
    speaker_id: 0
    length_scale: 0.95
    noise_scale: 0.667
    noise_w: 0.6
    sample_rate: 24000
tts_channels:
    chinese:
        engine: edge_tts
        executable_path: null
        model_path: null
        config_path: null
        voice_name: zh-TW-HsiaoChenNeural
        speaker_id: null
        length_scale: 0.95
        noise_scale: 0.667
        noise_w: 0.6
        sample_rate: 24000
    english:
        engine: edge_tts
        executable_path: null
        model_path: null
        config_path: null
        voice_name: en-US-JennyNeural
        speaker_id: null
        length_scale: 0.95
        noise_scale: 0.667
        noise_w: 0.6
        sample_rate: 24000
runtime:
    sample_rate: 24000
    chunk_ms: 40
    asr_queue_maxsize: 16
    llm_queue_maxsize: 4
    tts_queue_maxsize: 4
    translation_exact_cache_size: 256
    translation_prefix_min_delta_chars: 6
    tts_cancel_pending_on_new_final: true
    tts_cancel_policy: all_pending
    tts_max_wait_ms: 2500
    tts_max_chars: 140
    tts_drop_backlog_threshold: 3
    llm_streaming_tokens: 12
    max_pipeline_latency_ms: 2200
    local_echo_guard_enabled: true
    local_echo_guard_resume_delay_ms: 300
    remote_echo_guard_resume_delay_ms: 300
    config_schema_version: 3
    last_migration_note: ''
    warmup_on_start: true
    asr_queue_maxsize_chinese: 16
    asr_queue_maxsize_english: 12
    llm_queue_maxsize_zh_to_en: 4
    llm_queue_maxsize_en_to_zh: 4
    tts_queue_maxsize_chinese: 4
    tts_queue_maxsize_english: 4
health_last_success:
    asr: ''
    llm: ''
    tts: ''
chinese_tts:
    engine: edge_tts
    executable_path: ''
    model_path: ''
    config_path: ''
    voice_name: zh-TW-HsiaoChenNeural
    speaker_id: 0
    length_scale: 0.95
    noise_scale: 0.667
    noise_w: 0.6
    sample_rate: 24000
english_tts:
    engine: edge_tts
    executable_path: ''
    model_path: ''
    config_path: ''
    voice_name: en-US-JennyNeural
    speaker_id: 0
    length_scale: 0.95
    noise_scale: 0.667
    noise_w: 0.6
    sample_rate: 24000
"""


def _runtime_base_dirs() -> list[Path]:
    dirs: list[Path] = [Path.cwd()]

    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(item)
    return unique_dirs


def _resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in _runtime_base_dirs():
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def _resolve_write_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    existing = _resolve_existing_path(path)
    if existing.exists():
        return existing
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / path
    return Path.cwd() / path


def _ensure_config_file(config_path: str | Path = "config.yaml") -> Path:
    target = _resolve_write_path(config_path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(EMBEDDED_DEFAULT_CONFIG_YAML, encoding="utf-8")
    return target


def load_config(config_path: str | Path = "config.yaml", fallback_path: str | Path = "config.example.yaml") -> AppConfig:
    path = _resolve_existing_path(config_path)
    if not path.exists():
        path = _ensure_config_file(config_path)

    with path.open("r", encoding="utf-8") as fp:
        raw: dict[str, Any] = yaml.safe_load(fp) or {}
    raw = _normalize_external_config_keys(raw)

    migrated = migrate_legacy_config(raw)
    config = AppConfig.from_dict(migrated)
    return config


def save_config(config: AppConfig, config_path: str | Path = "config.yaml") -> Path:
    path = _resolve_write_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _present_external_config_keys(config.to_dict())
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
    return path


def _normalize_external_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(raw)
    asr_channels = data.get("asr_channels")
    if isinstance(asr_channels, dict):
        if "chinese" in asr_channels and "local" not in asr_channels:
            asr_channels["local"] = asr_channels.get("chinese")
        if "english" in asr_channels and "remote" not in asr_channels:
            asr_channels["remote"] = asr_channels.get("english")
    llm_channels = data.get("llm_channels")
    if isinstance(llm_channels, dict):
        if "zh_to_en" in llm_channels and "local" not in llm_channels:
            llm_channels["local"] = llm_channels.get("zh_to_en")
        if "en_to_zh" in llm_channels and "remote" not in llm_channels:
            llm_channels["remote"] = llm_channels.get("en_to_zh")
    tts_channels = data.get("tts_channels")
    if isinstance(tts_channels, dict):
        if "chinese" in tts_channels and "local" not in tts_channels:
            tts_channels["local"] = tts_channels.get("chinese")
        if "english" in tts_channels and "remote" not in tts_channels:
            tts_channels["remote"] = tts_channels.get("english")
    if "chinese_tts" in data and "meeting_tts" not in data:
        data["meeting_tts"] = data.get("chinese_tts")
    if "english_tts" in data and "local_tts" not in data:
        data["local_tts"] = data.get("english_tts")
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        mapping = {
            "asr_queue_maxsize_chinese": "asr_queue_maxsize_local",
            "asr_queue_maxsize_english": "asr_queue_maxsize_remote",
            "llm_queue_maxsize_zh_to_en": "llm_queue_maxsize_local",
            "llm_queue_maxsize_en_to_zh": "llm_queue_maxsize_remote",
            "tts_queue_maxsize_chinese": "tts_queue_maxsize_local",
            "tts_queue_maxsize_english": "tts_queue_maxsize_remote",
        }
        for src, dst in mapping.items():
            if src in runtime and dst not in runtime:
                runtime[dst] = runtime.get(src)
        for src in mapping:
            runtime.pop(src, None)
    return data


def _present_external_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(raw)
    asr_channels = data.get("asr_channels")
    if isinstance(asr_channels, dict):
        asr_channels["chinese"] = asr_channels.get("local")
        asr_channels["english"] = asr_channels.get("remote")
        asr_channels.pop("local", None)
        asr_channels.pop("remote", None)
    llm_channels = data.get("llm_channels")
    if isinstance(llm_channels, dict):
        llm_channels["zh_to_en"] = llm_channels.get("local")
        llm_channels["en_to_zh"] = llm_channels.get("remote")
        llm_channels.pop("local", None)
        llm_channels.pop("remote", None)
    tts_channels = data.get("tts_channels")
    if isinstance(tts_channels, dict):
        tts_channels["chinese"] = tts_channels.get("local")
        tts_channels["english"] = tts_channels.get("remote")
        tts_channels.pop("local", None)
        tts_channels.pop("remote", None)
    if "meeting_tts" in data:
        data["chinese_tts"] = data.pop("meeting_tts")
    if "local_tts" in data:
        data["english_tts"] = data.pop("local_tts")
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        runtime["asr_queue_maxsize_chinese"] = runtime.get("asr_queue_maxsize_local")
        runtime["asr_queue_maxsize_english"] = runtime.get("asr_queue_maxsize_remote")
        runtime["llm_queue_maxsize_zh_to_en"] = runtime.get("llm_queue_maxsize_local")
        runtime["llm_queue_maxsize_en_to_zh"] = runtime.get("llm_queue_maxsize_remote")
        runtime["tts_queue_maxsize_chinese"] = runtime.get("tts_queue_maxsize_local")
        runtime["tts_queue_maxsize_english"] = runtime.get("tts_queue_maxsize_remote")
        runtime.pop("asr_queue_maxsize_local", None)
        runtime.pop("asr_queue_maxsize_remote", None)
        runtime.pop("llm_queue_maxsize_local", None)
        runtime.pop("llm_queue_maxsize_remote", None)
        runtime.pop("tts_queue_maxsize_local", None)
        runtime.pop("tts_queue_maxsize_remote", None)
    return data


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not is_legacy_config(raw):
        return deepcopy(raw)

    result: dict[str, Any] = AppConfig().to_dict()

    audio = raw.get("audio") or {}
    result["audio"]["meeting_in"] = str(audio.get("meeting_in") or audio.get("remote_in") or "")
    result["audio"]["microphone_in"] = str(audio.get("microphone_in") or audio.get("local_mic_in") or "")
    result["audio"]["speaker_out"] = str(audio.get("speaker_out") or audio.get("local_tts_out") or "")
    result["audio"]["meeting_out"] = str(audio.get("meeting_out") or audio.get("meeting_tts_out") or "")

    mode = str((raw.get("direction") or {}).get("mode", "") or raw.get("session_mode", "meeting_to_local"))
    mode_map = {
        "remote_only": "meeting_to_local",
        "local_only": "local_to_meeting",
        "bidirectional": "bidirectional",
    }
    result["direction"]["mode"] = mode_map.get(mode, mode if mode in mode_map.values() else "meeting_to_local")

    language = raw.get("language") or {}
    result["language"]["meeting_source"] = str(language.get("meeting_source") or language.get("remote_source") or "en")
    result["language"]["meeting_target"] = str(language.get("meeting_target") or language.get("remote_target") or "zh-TW")
    result["language"]["local_source"] = str(language.get("local_source") or "zh-TW")
    result["language"]["local_target"] = str(language.get("local_target") or "en")

    asr = raw.get("asr") or {}
    openai = raw.get("openai") or {}
    result["asr"]["model"] = str(asr.get("model") or openai.get("asr_model") or "large-v3")
    result["asr"]["device"] = str(asr.get("device") or "cuda")
    result["asr"]["compute_type"] = str(asr.get("compute_type") or "float16")
    result["asr"]["beam_size"] = int(asr.get("beam_size", 1))
    result["asr"]["condition_on_previous_text"] = bool(asr.get("condition_on_previous_text", True))
    result["asr"]["temperature_fallback"] = str(asr.get("temperature_fallback", result["asr"]["temperature_fallback"]))
    result["asr"]["no_speech_threshold"] = float(asr.get("no_speech_threshold", result["asr"]["no_speech_threshold"]))
    if isinstance(asr.get("vad"), dict):
        result["asr"]["vad"].update(asr["vad"])
    if isinstance(asr.get("streaming"), dict):
        result["asr"]["streaming"].update(asr["streaming"])
    result["asr_channels"]["local"] = deepcopy(result["asr"])
    result["asr_channels"]["remote"] = deepcopy(result["asr"])
    result["asr_channels"]["local"]["temperature_fallback"] = "0.0,0.2"
    result["asr_channels"]["remote"]["temperature_fallback"] = "0.0,0.2,0.4"
    result["asr_channels"]["local"]["no_speech_threshold"] = 0.65
    result["asr_channels"]["remote"]["no_speech_threshold"] = 0.55

    llm = raw.get("llm") or {}
    result["llm"]["backend"] = "lm_studio"
    llm_base_url = str(llm.get("base_url", "")).strip()
    legacy_base_url = str(openai.get("base_url", "")).strip()
    if llm_base_url:
        result["llm"]["base_url"] = llm_base_url
    if legacy_base_url and "127.0.0.1" in legacy_base_url:
        result["llm"]["base_url"] = legacy_base_url
    if (not str(result["llm"]["base_url"]).strip()) or ("11434" in str(result["llm"]["base_url"])):
        result["llm"]["base_url"] = "http://127.0.0.1:1234"
    result["llm"]["model"] = str(llm.get("model") or openai.get("translate_model") or result["llm"]["model"])
    result["llm"]["temperature"] = float(llm.get("temperature", result["llm"]["temperature"]))
    result["llm"]["top_p"] = float(llm.get("top_p", result["llm"]["top_p"]))
    result["llm"]["max_output_tokens"] = int(llm.get("max_output_tokens", result["llm"]["max_output_tokens"]))
    result["llm"]["repeat_penalty"] = float(llm.get("repeat_penalty", result["llm"]["repeat_penalty"]))
    result["llm"]["stop_tokens"] = str(llm.get("stop_tokens", result["llm"]["stop_tokens"]))
    result["llm"]["request_timeout_sec"] = int(llm.get("request_timeout_sec", result["llm"]["request_timeout_sec"]))
    if isinstance(llm.get("sliding_window"), dict):
        result["llm"]["sliding_window"].update(llm["sliding_window"])
    if isinstance(llm.get("profiles"), dict):
        for key in ("live_caption_fast", "live_caption_stable", "speech_output_natural", "technical_meeting"):
            profile_raw = (llm.get("profiles") or {}).get(key)
            if isinstance(profile_raw, dict):
                result["llm"]["profiles"][key].update(profile_raw)
    result["llm"]["caption_profile"] = str(llm.get("caption_profile", result["llm"]["caption_profile"]))
    result["llm"]["speech_profile"] = str(llm.get("speech_profile", result["llm"]["speech_profile"]))
    result["llm_channels"]["local"] = deepcopy(result["llm"])
    result["llm_channels"]["remote"] = deepcopy(result["llm"])
    result["llm_channels"]["local"]["max_output_tokens"] = 96
    result["llm_channels"]["remote"]["max_output_tokens"] = 160
    result["llm_channels"]["local"]["stop_tokens"] = "</target>,Translation:"
    result["llm_channels"]["remote"]["stop_tokens"] = "</target>,翻譯:"

    tts = raw.get("tts") or {}
    result["tts"]["executable_path"] = str(tts.get("executable_path", result["tts"]["executable_path"]))
    result["tts"]["model_path"] = str(tts.get("model_path", result["tts"]["model_path"]))
    result["tts"]["config_path"] = str(tts.get("config_path", result["tts"]["config_path"]))
    result["tts"]["voice_name"] = str(tts.get("voice_name", result["tts"]["voice_name"]))
    result["tts"]["speaker_id"] = int(tts.get("speaker_id", result["tts"]["speaker_id"]))
    result["tts"]["length_scale"] = float(tts.get("length_scale", result["tts"]["length_scale"]))
    result["tts"]["noise_scale"] = float(tts.get("noise_scale", result["tts"]["noise_scale"]))
    result["tts"]["noise_w"] = float(tts.get("noise_w", result["tts"]["noise_w"]))
    result["tts"]["sample_rate"] = int(tts.get("sample_rate", result["tts"]["sample_rate"]))
    result["meeting_tts"] = deepcopy(result["tts"])
    result["local_tts"] = deepcopy(result["tts"])
    result["meeting_tts"]["voice_name"] = str(result["meeting_tts"].get("voice_name") or "zh-TW-HsiaoChenNeural")
    result["local_tts"]["voice_name"] = str(result["local_tts"].get("voice_name") or "en-US-JennyNeural")
    if str(result["local_tts"]["voice_name"]).strip().lower().startswith("zh-"):
        result["local_tts"]["voice_name"] = "en-US-JennyNeural"
    result["tts_channels"]["local"] = deepcopy(result["meeting_tts"])
    result["tts_channels"]["remote"] = deepcopy(result["local_tts"])

    runtime = raw.get("runtime") or {}
    result["runtime"]["sample_rate"] = int(runtime.get("sample_rate", raw.get("sample_rate", 48000)))
    result["runtime"]["chunk_ms"] = int(runtime.get("chunk_ms", raw.get("chunk_ms", 100)))
    asr_shared = int(runtime.get("asr_queue_maxsize", result["runtime"]["asr_queue_maxsize"]))
    llm_shared = int(runtime.get("llm_queue_maxsize", result["runtime"]["llm_queue_maxsize"]))
    tts_shared = int(runtime.get("tts_queue_maxsize", result["runtime"]["tts_queue_maxsize"]))
    result["runtime"]["asr_queue_maxsize_local"] = int(runtime.get("asr_queue_maxsize_local", asr_shared))
    result["runtime"]["asr_queue_maxsize_remote"] = int(runtime.get("asr_queue_maxsize_remote", asr_shared))
    result["runtime"]["llm_queue_maxsize_local"] = int(runtime.get("llm_queue_maxsize_local", llm_shared))
    result["runtime"]["llm_queue_maxsize_remote"] = int(runtime.get("llm_queue_maxsize_remote", llm_shared))
    result["runtime"]["tts_queue_maxsize_local"] = int(runtime.get("tts_queue_maxsize_local", tts_shared))
    result["runtime"]["tts_queue_maxsize_remote"] = int(runtime.get("tts_queue_maxsize_remote", tts_shared))
    result["runtime"]["asr_queue_maxsize"] = result["runtime"]["asr_queue_maxsize_local"]
    result["runtime"]["llm_queue_maxsize"] = result["runtime"]["llm_queue_maxsize_local"]
    result["runtime"]["tts_queue_maxsize"] = result["runtime"]["tts_queue_maxsize_local"]
    result["runtime"]["translation_exact_cache_size"] = int(
        runtime.get("translation_exact_cache_size", result["runtime"]["translation_exact_cache_size"])
    )
    result["runtime"]["translation_prefix_min_delta_chars"] = int(
        runtime.get("translation_prefix_min_delta_chars", result["runtime"]["translation_prefix_min_delta_chars"])
    )
    result["runtime"]["tts_cancel_pending_on_new_final"] = bool(
        runtime.get("tts_cancel_pending_on_new_final", result["runtime"]["tts_cancel_pending_on_new_final"])
    )
    result["runtime"]["tts_cancel_policy"] = str(runtime.get("tts_cancel_policy", result["runtime"]["tts_cancel_policy"]))
    result["runtime"]["tts_max_wait_ms"] = int(runtime.get("tts_max_wait_ms", result["runtime"]["tts_max_wait_ms"]))
    result["runtime"]["tts_max_chars"] = int(runtime.get("tts_max_chars", result["runtime"]["tts_max_chars"]))
    result["runtime"]["tts_drop_backlog_threshold"] = int(
        runtime.get("tts_drop_backlog_threshold", result["runtime"]["tts_drop_backlog_threshold"])
    )
    result["runtime"]["llm_streaming_tokens"] = int(runtime.get("llm_streaming_tokens", result["runtime"]["llm_streaming_tokens"]))
    result["runtime"]["max_pipeline_latency_ms"] = int(runtime.get("max_pipeline_latency_ms", result["runtime"]["max_pipeline_latency_ms"]))
    result["runtime"]["local_echo_guard_enabled"] = bool(
        runtime.get("local_echo_guard_enabled", result["runtime"]["local_echo_guard_enabled"])
    )
    result["runtime"]["local_echo_guard_resume_delay_ms"] = int(
        runtime.get("local_echo_guard_resume_delay_ms", result["runtime"]["local_echo_guard_resume_delay_ms"])
    )
    result["runtime"]["remote_echo_guard_resume_delay_ms"] = int(
        runtime.get("remote_echo_guard_resume_delay_ms", result["runtime"]["remote_echo_guard_resume_delay_ms"])
    )
    result["runtime"]["config_schema_version"] = 3
    result["runtime"]["last_migration_note"] = "migrated_from_legacy"
    result["runtime"]["warmup_on_start"] = bool(runtime.get("warmup_on_start", result["runtime"]["warmup_on_start"]))

    legacy_test = raw.get("provider_test_last_success") or {}
    health = raw.get("health_last_success") or {}
    result["health_last_success"]["asr"] = str(health.get("asr") or legacy_test.get("asr") or "")
    result["health_last_success"]["llm"] = str(health.get("llm") or legacy_test.get("translate") or "")
    result["health_last_success"]["tts"] = str(health.get("tts") or legacy_test.get("tts") or "")
    return result


def is_legacy_config(raw: dict[str, Any]) -> bool:
    if "openai" in raw or "model" in raw or "session_mode" in raw:
        return True
    if (
        "direction" not in raw
        or "asr" not in raw
        or "llm" not in raw
        or "tts" not in raw
        or "runtime" not in raw
    ):
        return True
    audio = raw.get("audio") or {}
    if any(key in audio for key in ("remote_in", "local_mic_in", "local_tts_out", "meeting_tts_out")):
        return True
    return False
