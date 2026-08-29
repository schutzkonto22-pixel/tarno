"""System tray icon for the TARNO control UI."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


class TrayIcon(QSystemTrayIcon):
    """Tray icon with context menu."""

    open_requested = Signal()
    mute_requested = Signal(bool)
    start_stop_requested = Signal()
    settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip("TARNO")

        self._menu = QMenu(parent)

        self._open_action = QAction("Öffnen")
        self._open_action.triggered.connect(self.open_requested.emit)
        self._menu.addAction(self._open_action)

        self._menu.addSeparator()

        self._mute_action = QAction("Stumm")
        self._mute_action.setCheckable(True)
        self._mute_action.toggled.connect(self.mute_requested.emit)
        self._menu.addAction(self._mute_action)

        self._start_stop_action = QAction("Start")
        self._start_stop_action.triggered.connect(self.start_stop_requested.emit)
        self._menu.addAction(self._start_stop_action)

        self._settings_action = QAction("Einstellungen")
        self._settings_action.triggered.connect(self.settings_requested.emit)
        self._menu.addAction(self._settings_action)

        self._menu.addSeparator()

        self._quit_action = QAction("Beenden")
        self._quit_action.triggered.connect(self.quit_requested.emit)
        self._menu.addAction(self._quit_action)

        self.setContextMenu(self._menu)
        self.activated.connect(self._on_activated)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_requested.emit()

    def set_mute(self, muted: bool) -> None:
        self._mute_action.setChecked(muted)

    def set_engine_running(self, running: bool) -> None:
        self._start_stop_action.setText("Stop" if running else "Start")

    def show_icon(self) -> None:
        self.show()
