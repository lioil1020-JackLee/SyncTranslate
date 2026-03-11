from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app.schemas import AppConfig


class ModelsPage(QWidget):
    def __init__(
        self,
        on_test_asr: Callable[[], None],
        on_test_translate: Callable[[], None],
        on_test_tts: Callable[[], None],
        on_cancel_test: Callable[[], None],
        on_clear_test_state: Callable[[], None],
    ) -> None:
        super().__init__()
        self.remote_source_combo = QComboBox()
        self.remote_target_combo = QComboBox()
        self.local_source_combo = QComboBox()
        self.local_target_combo = QComboBox()
        self.asr_combo = QComboBox()
        self.translate_combo = QComboBox()
        self.tts_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.openai_api_key_env_edit = QLineEdit()
        self.openai_base_url_edit = QLineEdit()
        self.openai_asr_model_edit = QLineEdit()
        self.openai_translate_model_edit = QLineEdit()
        self.openai_tts_model_edit = QLineEdit()
        self.openai_tts_voice_edit = QLineEdit()
        self.runtime_status_label = QLabel("Runtime status: -")
        self.test_status_label = QLabel("Provider test: -")
        self.last_success_label = QLabel("Last success: -")
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
        self.translate_combo.addItem("mock", "mock")
        self.translate_combo.addItem("openai", "openai")
        self.tts_combo.addItem("mock", "mock")
        self.tts_combo.addItem("openai", "openai")
        self.mode_combo.addItems(["快譯模式", "穩譯模式"])

        form = QFormLayout()
        form.addRow("remote source", self.remote_source_combo)
        form.addRow("remote target", self.remote_target_combo)
        form.addRow("local source", self.local_source_combo)
        form.addRow("local target", self.local_target_combo)
        form.addRow("ASR", self.asr_combo)
        form.addRow("Translate", self.translate_combo)
        form.addRow("TTS", self.tts_combo)
        form.addRow("翻譯模式", self.mode_combo)
        form.addRow("OpenAI API key env", self.openai_api_key_env_edit)
        form.addRow("OpenAI base URL", self.openai_base_url_edit)
        form.addRow("OpenAI ASR model", self.openai_asr_model_edit)
        form.addRow("OpenAI translate model", self.openai_translate_model_edit)
        form.addRow("OpenAI TTS model", self.openai_tts_model_edit)
        form.addRow("OpenAI TTS voice", self.openai_tts_voice_edit)

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

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("模型與語言設定（第 6 階段骨架）"))
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.runtime_status_label)
        layout.addWidget(self.test_status_label)
        layout.addWidget(self.last_success_label)
        layout.addStretch(1)

    def apply_config(self, config: AppConfig) -> None:
        self._select_by_text(self.remote_source_combo, config.language.remote_source)
        self._select_by_text(self.remote_target_combo, config.language.remote_target)
        self._select_by_text(self.local_source_combo, config.language.local_source)
        self._select_by_text(self.local_target_combo, config.language.local_target)
        self._select_by_data(self.asr_combo, config.model.asr_provider)
        self._select_by_data(self.translate_combo, config.model.translate_provider)
        self._select_by_data(self.tts_combo, config.model.tts_provider)
        self.openai_api_key_env_edit.setText(config.openai.api_key_env)
        self.openai_base_url_edit.setText(config.openai.base_url)
        self.openai_asr_model_edit.setText(config.openai.asr_model)
        self.openai_translate_model_edit.setText(config.openai.translate_model)
        self.openai_tts_model_edit.setText(config.openai.tts_model)
        self.openai_tts_voice_edit.setText(config.openai.tts_voice)

    def update_config(self, config: AppConfig) -> None:
        config.language.remote_source = self.remote_source_combo.currentText()
        config.language.remote_target = self.remote_target_combo.currentText()
        config.language.local_source = self.local_source_combo.currentText()
        config.language.local_target = self.local_target_combo.currentText()
        config.model.asr_provider = str(self.asr_combo.currentData())
        config.model.translate_provider = str(self.translate_combo.currentData())
        config.model.tts_provider = str(self.tts_combo.currentData())
        config.openai.api_key_env = self.openai_api_key_env_edit.text().strip() or "OPENAI_API_KEY"
        config.openai.base_url = self.openai_base_url_edit.text().strip() or "https://api.openai.com/v1"
        config.openai.asr_model = self.openai_asr_model_edit.text().strip() or "gpt-4o-mini-transcribe"
        config.openai.translate_model = self.openai_translate_model_edit.text().strip() or "gpt-4.1-mini"
        config.openai.tts_model = self.openai_tts_model_edit.text().strip() or "gpt-4o-mini-tts"
        config.openai.tts_voice = self.openai_tts_voice_edit.text().strip() or "alloy"

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
