# Testing

## Automated test commands

完整測試目前以 `unittest discover` 為主：

```powershell
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

若你的環境已安裝 `pytest`，也可以用：

```powershell
pytest -q
```

system check：

```powershell
uv run python .\main.py --check
```

## Coverage focus

目前測試主要覆蓋：

- audio router policy
- bidirectional runtime behavior
- per-direction ASR / LLM selection
- remote / local translation enable switches
- TTS style policy
- transcript buffering
- session lifecycle
- config migration and canonical save format
- diagnostics export
- live caption export
- UI labels and behavior
- fake end-to-end flow from ASR -> translation -> subtitles -> TTS

代表性測試檔：

- [`tests/test_audio_router_policy.py`](/e:/py/SyncTranslate/tests/test_audio_router_policy.py)
- [`tests/test_pipeline_integration.py`](/e:/py/SyncTranslate/tests/test_pipeline_integration.py)
- [`tests/test_multilingual_channel_policy.py`](/e:/py/SyncTranslate/tests/test_multilingual_channel_policy.py)
- [`tests/test_config_migration.py`](/e:/py/SyncTranslate/tests/test_config_migration.py)
- [`tests/test_session_service.py`](/e:/py/SyncTranslate/tests/test_session_service.py)

## Manual smoke checklist

1. 啟動 LM Studio，載入 `llm_channels.local` 與 `llm_channels.remote` 要用的模型。
2. 開啟程式，確認四個 audio route 都已選好。
3. 執行 system check，確認 ASR / LLM / TTS 都 ready。
4. 啟動 live session。
5. 對麥克風說話，確認：
   - `local_original` 更新
   - `local_translated` 依設定更新
   - 若本地方向設為 `tts`，語音會送往 `meeting_out`
   - 若本地方向設為 `passthrough`，原始麥克風音訊會送往 `meeting_out`
6. 餵入會議音訊，確認：
   - `meeting_original` 更新
   - `meeting_translated` 依設定更新
   - 若遠端方向設為 `tts`，語音會送往 `speaker_out`
   - 若遠端方向設為 `passthrough`，原始會議音訊會送往 `speaker_out`
7. 分別調整兩個方向的 target language / TTS voice，確認 UI 會推導出正確的 output mode：
   - target language = `none` -> `passthrough`
   - target language 有值、voice = `none` -> `subtitle_only`
   - target language 有值、voice 有值 -> `tts`
8. 匯出四個字幕面板，確認檔案內容與 UI 一致。

## Expected steady-state rules

- Runtime 一律以 bidirectional 模式運作。
- 兩個 source 都會被啟動。
- `passthrough` 不會停用 ASR。
- remote / local 翻譯可獨立開關。
- remote / local ASR 與 LLM 設定可獨立選擇。
- translated panel 在翻譯關閉時仍會顯示 ASR 原文。
