from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class LiveCaptionPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.remote_original = QTextEdit()
        self.remote_translated = QTextEdit()
        self.local_original = QTextEdit()
        self.local_translated = QTextEdit()

        for editor in (
            self.remote_original,
            self.remote_translated,
            self.local_original,
            self.local_translated,
        ):
            editor.setReadOnly(True)

        grid = QGridLayout()
        grid.addWidget(QLabel("Remote Original"), 0, 0)
        grid.addWidget(QLabel("Remote Translated"), 0, 1)
        grid.addWidget(self.remote_original, 1, 0)
        grid.addWidget(self.remote_translated, 1, 1)
        grid.addWidget(QLabel("Local Original"), 2, 0)
        grid.addWidget(QLabel("Local Translated"), 2, 1)
        grid.addWidget(self.local_original, 3, 0)
        grid.addWidget(self.local_translated, 3, 1)

        clear_btn = QPushButton("清空字幕")
        clear_btn.clicked.connect(self.clear)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(clear_btn)

    def clear(self) -> None:
        self.remote_original.clear()
        self.remote_translated.clear()
        self.local_original.clear()
        self.local_translated.clear()

    def set_remote_original_lines(self, lines: list[str]) -> None:
        self.remote_original.setPlainText("\n".join(lines))

    def set_remote_translated_lines(self, lines: list[str]) -> None:
        self.remote_translated.setPlainText("\n".join(lines))

    def set_local_original_lines(self, lines: list[str]) -> None:
        self.local_original.setPlainText("\n".join(lines))

    def set_local_translated_lines(self, lines: list[str]) -> None:
        self.local_translated.setPlainText("\n".join(lines))
