"""Microphone / mute button widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QWidget


class MicButton(QPushButton):
    """Toggle button for voice/mute."""

    toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("🎙", parent)
        self.setObjectName("micButton")
        self.setCheckable(True)
        self.setChecked(True)
        self.setToolTip("Spracherkennung aktivieren/deaktivieren")
        self.clicked.connect(self._on_clicked)
        self.set_state(True)

    def _on_clicked(self, checked: bool) -> None:
        self.set_state(checked)
        self.toggled.emit(checked)

    def set_state(self, active: bool) -> None:
        self.setChecked(active)
        self.setStyleSheet(
            "border-color: #2dd4bf; background-color: rgba(45, 212, 191, 0.12);" if active
            else "border-color: #ef4444; background-color: rgba(239, 68, 68, 0.12);"
        )
        self.setToolTip("Spracherkennung deaktivieren" if active else "Spracherkennung aktivieren")

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(44)
        size.setHeight(44)
        return size
