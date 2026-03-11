from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.env_vars import list_env_var_names
from app.schemas import AppConfig


class ModelsPage(QWidget):
    API_BASE_URL_OPTIONS = [
        "https://api.openai.com/v1",
        "https://api.groq.com/openai/v1",
        "https://router.huggingface.co/v1",
    ]
    ASR_MODEL_OPTIONS = [
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "whisper-1",
        "whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "small",
        "medium",
        "large-v3",
    ]
    TRANSLATE_MODEL_OPTIONS = [
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
        "llama-3.3-70b-versatile",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]
    TTS_MODEL_OPTIONS = [
        "gpt-4o-mini-tts",
        "tts-1",
        "tts-1-hd",
    ]
    TTS_VOICE_OPTIONS = [
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
        "zh-TW-HsiaoChenNeural",
        "zh-TW-YunJheNeural",
        "en-US-AvaMultilingualNeural",
        "en-US-AndrewMultilingualNeural",
    ]

    def __init__(
        self,
        on_test_asr: Callable[[], None],
        on_test_translate: Callable[[], None],
        on_test_tts: Callable[[], None],
        on_cancel_test: Callable[[], None],
        on_clear_test_state: Callable[[], None],
        on_settings_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_settings_changed = on_settings_changed
        self._suspend_notify = False

        self.remote_source_combo = QComboBox()
        self.remote_target_combo = QComboBox()
        self.local_source_combo = QComboBox()
        self.local_target_combo = QComboBox()
        self.asr_combo = QComboBox()
        self.translate_combo = QComboBox()
        self.tts_combo = QComboBox()
        self.mode_combo = QComboBox()

        self.openai_api_key_env_combo = self._new_editable_combo(self._api_key_env_options())
        self.openai_base_url_combo = self._new_editable_combo(self.API_BASE_URL_OPTIONS)
        self.openai_asr_model_combo = self._new_editable_combo(self.ASR_MODEL_OPTIONS)
        self.openai_translate_model_combo = self._new_editable_combo(self.TRANSLATE_MODEL_OPTIONS)
        self.openai_tts_model_combo = self._new_editable_combo(self.TTS_MODEL_OPTIONS)
        self.openai_tts_voice_combo = self._new_editable_combo(self.TTS_VOICE_OPTIONS)

        self.runtime_status_label = QLabel("Runtime status: -")
        self.test_status_label = QLabel("Provider test: -")
        self.last_success_label = QLabel("Last success: -")

        self.quick_groq_btn = QPushButton("套用 Groq 免費方案")
        self.quick_hf_btn = QPushButton("套用 HF 免費方案")
        self.quick_openai_btn = QPushButton("套用 OpenAI 預設")

        self.test_asr_btn = QPushButton("測試 ASR")
        self.test_translate_btn = QPushButton("測試 Translate")
        self.test_tts_btn = QPushButton("測試 TTS")
        self.cancel_test_btn = QPushButton("取消測試")
        self.clear_test_state_btn = QPushButton("清除測試狀態")
        self.cancel_test_btn.setEnabled(False)

        for combo in (
            self.remote_source_combo,
            self.remote_target_combo,
            self.local_source_combo,
            self.local_target_combo,
        ):
            combo.addItems(["en", "zh-TW"])

        self.asr_combo.addItem("mock", "mock")
        self.asr_combo.addItem("openai", "openai")
        self.asr_combo.addItem("groq", "groq")
        self.asr_combo.addItem("huggingface", "huggingface")
        self.asr_combo.addItem("local", "local")

        self.translate_combo.addItem("mock", "mock")
        self.translate_combo.addItem("openai", "openai")
        self.translate_combo.addItem("groq", "groq")
        self.translate_combo.addItem("huggingface", "huggingface")

        self.tts_combo.addItem("mock", "mock")
        self.tts_combo.addItem("openai", "openai")
        self.tts_combo.addItem("edge_tts", "edge_tts")

        self.mode_combo.addItem("快譯模式", "fast")
        self.mode_combo.addItem("精譯模式", "quality")

        self.openai_api_key_env_combo.setToolTip("從系統環境變數讀取 API key 的變數名稱")
        self.openai_base_url_combo.setToolTip("OpenAI 相容 API 的 base URL")

        form = QFormLayout()
        form.addRow("remote source", self.remote_source_combo)
        form.addRow("remote target", self.remote_target_combo)
        form.addRow("local source", self.local_source_combo)
        form.addRow("local target", self.local_target_combo)
        form.addRow("ASR Provider", self.asr_combo)
        form.addRow("Translate Provider", self.translate_combo)
        form.addRow("TTS Provider", self.tts_combo)
        form.addRow("翻譯模式", self.mode_combo)
        form.addRow("API key env", self.openai_api_key_env_combo)
        form.addRow("API base URL", self.openai_base_url_combo)
        form.addRow("ASR model", self.openai_asr_model_combo)
        form.addRow("Translate model", self.openai_translate_model_combo)
        form.addRow("TTS model", self.openai_tts_model_combo)
        form.addRow("TTS voice", self.openai_tts_voice_combo)

        quick_buttons = QHBoxLayout()
        self.quick_groq_btn.clicked.connect(self._apply_groq_free_preset)
        self.quick_hf_btn.clicked.connect(self._apply_hf_free_preset)
        self.quick_openai_btn.clicked.connect(self._apply_openai_default_preset)
        quick_buttons.addWidget(self.quick_groq_btn)
        quick_buttons.addWidget(self.quick_hf_btn)
        quick_buttons.addWidget(self.quick_openai_btn)

        buttons = QHBoxLayout()
        self.test_asr_btn.clicked.connect(on_test_asr)
        self.test_translate_btn.clicked.connect(on_test_translate)
        self.test_tts_btn.clicked.connect(on_test_tts)
        self.cancel_test_btn.clicked.connect(on_cancel_test)
        self.clear_test_state_btn.clicked.connect(on_clear_test_state)
        buttons.addWidget(self.test_asr_btn)
        buttons.addWidget(self.test_translate_btn)
        buttons.addWidget(self.test_tts_btn)
        buttons.addWidget(self.cancel_test_btn)
        buttons.addWidget(self.clear_test_state_btn)

        for combo in (
            self.remote_source_combo,
            self.remote_target_combo,
            self.local_source_combo,
            self.local_target_combo,
            self.asr_combo,
            self.translate_combo,
            self.tts_combo,
            self.mode_combo,
            self.openai_api_key_env_combo,
            self.openai_base_url_combo,
            self.openai_asr_model_combo,
            self.openai_translate_model_combo,
            self.openai_tts_model_combo,
            self.openai_tts_voice_combo,
        ):
            combo.currentTextChanged.connect(self._notify_settings_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("模型與語言設定"))
        layout.addLayout(form)
        layout.addLayout(quick_buttons)
        layout.addLayout(buttons)
        layout.addWidget(self.runtime_status_label)
        layout.addWidget(self.test_status_label)
        layout.addWidget(self.last_success_label)
        layout.addStretch(1)

    def apply_config(self, config: AppConfig) -> None:
        self._suspend_notify = True
        try:
            self._refresh_api_key_env_options()
            self._select_by_text(self.remote_source_combo, config.language.remote_source)
            self._select_by_text(self.remote_target_combo, config.language.remote_target)
            self._select_by_text(self.local_source_combo, config.language.local_source)
            self._select_by_text(self.local_target_combo, config.language.local_target)
            self._select_by_data(self.asr_combo, config.model.asr_provider)
            self._select_by_data(self.translate_combo, config.model.translate_provider)
            self._select_by_data(self.tts_combo, config.model.tts_provider)
            self._select_by_data(self.mode_combo, config.translate_mode)

            self._set_editable_combo_value(self.openai_api_key_env_combo, config.openai.api_key_env)
            self._set_editable_combo_value(self.openai_base_url_combo, config.openai.base_url)
            self._set_editable_combo_value(self.openai_asr_model_combo, config.openai.asr_model)
            self._set_editable_combo_value(self.openai_translate_model_combo, config.openai.translate_model)
            self._set_editable_combo_value(self.openai_tts_model_combo, config.openai.tts_model)
            self._set_editable_combo_value(self.openai_tts_voice_combo, config.openai.tts_voice)
        finally:
            self._suspend_notify = False

    def update_config(self, config: AppConfig) -> None:
        config.language.remote_source = self.remote_source_combo.currentText()
        config.language.remote_target = self.remote_target_combo.currentText()
        config.language.local_source = self.local_source_combo.currentText()
        config.language.local_target = self.local_target_combo.currentText()
        config.model.asr_provider = str(self.asr_combo.currentData())
        config.model.translate_provider = str(self.translate_combo.currentData())
        config.model.tts_provider = str(self.tts_combo.currentData())
        config.translate_mode = str(self.mode_combo.currentData() or "fast")

        config.openai.api_key_env = self.openai_api_key_env_combo.currentText().strip() or "OPENAI_API_KEY"
        config.openai.base_url = self.openai_base_url_combo.currentText().strip() or "https://api.openai.com/v1"
        config.openai.asr_model = self.openai_asr_model_combo.currentText().strip() or "gpt-4o-transcribe"
        config.openai.translate_model = self.openai_translate_model_combo.currentText().strip() or "gpt-5-mini"
        config.openai.tts_model = self.openai_tts_model_combo.currentText().strip() or "gpt-4o-mini-tts"
        config.openai.tts_voice = self.openai_tts_voice_combo.currentText().strip() or "alloy"

    def set_runtime_status(self, text: str) -> None:
        self.runtime_status_label.setText(text)

    def set_test_status(self, text: str) -> None:
        self.test_status_label.setText(text)

    def set_test_running(self, running: bool) -> None:
        enabled = not running
        self.test_asr_btn.setEnabled(enabled)
        self.test_translate_btn.setEnabled(enabled)
        self.test_tts_btn.setEnabled(enabled)
        self.cancel_test_btn.setEnabled(running)

    def set_last_success(self, text: str) -> None:
        self.last_success_label.setText(text)

    def _apply_groq_free_preset(self) -> None:
        self._select_by_data(self.asr_combo, "groq")
        self._select_by_data(self.translate_combo, "groq")
        self._select_by_data(self.tts_combo, "edge_tts")
        self._set_editable_combo_value(self.openai_api_key_env_combo, "GROQ_API_KEY")
        self._set_editable_combo_value(self.openai_base_url_combo, "https://api.groq.com/openai/v1")
        self._set_editable_combo_value(self.openai_asr_model_combo, "whisper-large-v3")
        self._set_editable_combo_value(self.openai_translate_model_combo, "llama-3.3-70b-versatile")
        self._set_editable_combo_value(self.openai_tts_voice_combo, "zh-TW-HsiaoChenNeural")
        self.set_test_status("Provider test: Groq free preset applied (TTS=edge_tts).")
        self._notify_settings_changed()

    def _apply_hf_free_preset(self) -> None:
        self._select_by_data(self.asr_combo, "mock")
        self._select_by_data(self.translate_combo, "huggingface")
        self._select_by_data(self.tts_combo, "edge_tts")
        self._set_editable_combo_value(self.openai_api_key_env_combo, "HF_TOKEN")
        self._set_editable_combo_value(self.openai_base_url_combo, "https://router.huggingface.co/v1")
        self._set_editable_combo_value(self.openai_asr_model_combo, "openai/whisper-large-v3-turbo")
        self._set_editable_combo_value(self.openai_translate_model_combo, "meta-llama/Llama-3.3-70B-Instruct")
        self._set_editable_combo_value(self.openai_tts_voice_combo, "zh-TW-HsiaoChenNeural")
        self.set_test_status("Provider test: HuggingFace free preset applied (ASR=mock, Translate=huggingface, TTS=edge_tts).")
        self._notify_settings_changed()

    def _apply_openai_default_preset(self) -> None:
        self._select_by_data(self.asr_combo, "openai")
        self._select_by_data(self.translate_combo, "openai")
        self._select_by_data(self.tts_combo, "openai")
        self._set_editable_combo_value(self.openai_api_key_env_combo, "OPENAI_API_KEY")
        self._set_editable_combo_value(self.openai_base_url_combo, "https://api.openai.com/v1")
        self._set_editable_combo_value(self.openai_asr_model_combo, "gpt-4o-transcribe")
        self._set_editable_combo_value(self.openai_translate_model_combo, "gpt-5-mini")
        self._set_editable_combo_value(self.openai_tts_model_combo, "gpt-4o-mini-tts")
        self._set_editable_combo_value(self.openai_tts_voice_combo, "alloy")
        self.set_test_status("Provider test: OpenAI preset applied.")
        self._notify_settings_changed()

    @staticmethod
    def _select_by_text(combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _select_by_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _new_editable_combo(options: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(options)
        return combo

    @staticmethod
    def _set_editable_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
        combo.setEditText(value)

    @staticmethod
    def _api_key_env_options() -> list[str]:
        preferred = [
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        ]
        all_env_names = list_env_var_names()
        result: list[str] = []
        seen: set[str] = set()
        for name in preferred + all_env_names:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def _refresh_api_key_env_options(self) -> None:
        current_text = self.openai_api_key_env_combo.currentText()
        options = self._api_key_env_options()
        if current_text and current_text not in options:
            options.insert(0, current_text)
        self.openai_api_key_env_combo.blockSignals(True)
        self.openai_api_key_env_combo.clear()
        self.openai_api_key_env_combo.addItems(options)
        self.openai_api_key_env_combo.blockSignals(False)
        if current_text:
            self._set_editable_combo_value(self.openai_api_key_env_combo, current_text)

    def _notify_settings_changed(self) -> None:
        if self._suspend_notify:
            return
        if self._on_settings_changed:
            self._on_settings_changed()
