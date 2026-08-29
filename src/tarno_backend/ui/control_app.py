"""Main TARNO control application (Control Center)."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tarno_backend.ui.overlay_window import OverlayState, OverlayWindow

log = logging.getLogger(__name__)


class ControlCenter(QMainWindow):
    """Hauptfenster für Chat, Status-Orb und Kontrollen."""

    text_submitted = Signal(str)
    mic_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None, show_overlay: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("TARNO Control Center")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(self._stylesheet())

        self._overlay: OverlayWindow | None = None
        if show_overlay:
            self._overlay = OverlayWindow()
            self._overlay.show()

        self._central = QWidget()
        self.setCentralWidget(self._central)
        main_layout = QHBoxLayout(self._central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left: chat area
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self._chat_history = QTextEdit()
        self._chat_history.setReadOnly(True)
        self._chat_history.setFont(QFont("Segoe UI", 11))
        chat_layout.addWidget(self._chat_history)

        input_layout = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Befehl eingeben...")
        self._input.returnPressed.connect(self._on_submit)
        input_layout.addWidget(self._input)

        send_button = QPushButton("Senden")
        send_button.clicked.connect(self._on_submit)
        input_layout.addWidget(send_button)

        self._mic_button = QPushButton("🎤")
        self._mic_button.setCheckable(True)
        self._mic_button.clicked.connect(self._on_mic_toggle)
        input_layout.addWidget(self._mic_button)

        chat_layout.addLayout(input_layout)
        splitter.addWidget(chat_panel)

        # Right: status / info panel
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("Bereit")
        self._status_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        info_layout.addWidget(self._status_label)

        self._provider_label = QLabel("Provider: -")
        info_layout.addWidget(self._provider_label)

        info_layout.addStretch()
        splitter.addWidget(info_panel)
        splitter.setSizes([700, 200])

    def _stylesheet(self) -> str:
        return """
        QMainWindow {
            background-color: #0b0f19;
            color: #c8e6ff;
        }
        QTextEdit {
            background-color: #111827;
            color: #c8e6ff;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 8px;
        }
        QLineEdit {
            background-color: #111827;
            color: #c8e6ff;
            border: 1px solid #1f2937;
            border-radius: 6px;
            padding: 6px;
        }
        QPushButton {
            background-color: #1f2937;
            color: #c8e6ff;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #374151;
        }
        QLabel {
            color: #c8e6ff;
        }
        """

    def _on_submit(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.add_message("user", text)
        self.text_submitted.emit(text)
        self._input.clear()

    def _on_mic_toggle(self, checked: bool) -> None:
        self.mic_toggled.emit(checked)
        self._mic_button.setText("🎤 An" if checked else "🎤")

    def add_message(self, role: str, text: str) -> None:
        label = {"user": "Sie", "assistant": "TARNO"}.get(role, role.capitalize())
        color = "#60a5fa" if role == "user" else "#a78bfa"
        self._chat_history.append(
            f'<span style="color:{color}; font-weight:bold">{label}:</span> {text}'
        )

    def set_status(self, text: str, overlay_state: OverlayState = OverlayState.IDLE) -> None:
        self._status_label.setText(text)
        if self._overlay is not None:
            self._overlay.set_state(overlay_state, text)

    def set_provider(self, name: str) -> None:
        self._provider_label.setText(f"Provider: {name}")

    def closeEvent(self, event: Any) -> None:
        if self._overlay is not None:
            self._overlay.close()
        super().closeEvent(event)


def launch_control_center() -> None:
    app = QApplication.instance() or QApplication([])
    window = ControlCenter()
    window.show()
    app.exec()
