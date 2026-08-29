"""GUI application entry point for the TARNO control UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QStyle, QSystemTrayIcon

from tarno_backend.core.config import TarnoConfig
from tarno_backend.ui.bus_listener import BusListener
from tarno_backend.ui.control_window import ControlWindow
from tarno_backend.ui.engine_controller import EngineController
from tarno_backend.ui.settings_dialog import SettingsDialog
from tarno_backend.ui.theme import get_stylesheet
from tarno_backend.ui.tray import TrayIcon

log = logging.getLogger(__name__)


def _get_default_icon() -> QIcon:
    """Return a default icon from the system style."""
    app = QApplication.instance()
    if app is None:
        return QIcon()
    style = app.style()
    if style is None:
        return QIcon()
    icon = style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    return icon if icon is not None else QIcon()


def _user_config_path() -> Path:
    return Path.home() / ".tarno" / "config" / "tarno_config.yaml"


def run_ui(config: TarnoConfig) -> None:
    """Create and run the TARNO control UI."""
    app = QApplication(sys.argv)
    app.setApplicationName("TARNO Control")
    app.setApplicationVersion("0.2.0")
    app.setStyleSheet(get_stylesheet())
    app.setWindowIcon(_get_default_icon())
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning("Kein System-Tray verfügbar")

    ui_config = config.ui
    save_path = _user_config_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    controller = EngineController()
    listener = BusListener()
    window = ControlWindow()
    # The permanent hologram overlay is disabled because it blocks the desktop.
    # OverlayWindow (tarno_backend.ui.overlay_window) is kept for a future
    # optional boot/loading animation.
    tray = TrayIcon()
    tray.setIcon(app.windowIcon())

    def open_settings() -> None:
        dialog = SettingsDialog(config, save_path, window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config.ui = dialog.get_config()
            try:
                config.save(save_path)
            except Exception as exc:
                log.exception("Einstellungen konnten nicht gespeichert werden")
                window.append_log(f"[ERROR] Einstellungen speichern fehlgeschlagen: {exc}")

    def toggle_engine() -> None:
        if controller.is_running():
            controller.stop_engine()
        else:
            controller.start_engine()

    # Wire UI actions to controller
    window.start_requested.connect(controller.start_engine)
    window.stop_requested.connect(controller.stop_engine)
    window.mute_requested.connect(controller.toggle_mute)
    window.text_submitted.connect(controller.send_text)
    window.settings_requested.connect(open_settings)

    tray.open_requested.connect(window.showNormal)
    tray.mute_requested.connect(controller.toggle_mute)
    tray.mute_requested.connect(window.set_mic_active)
    tray.start_stop_requested.connect(toggle_engine)
    tray.settings_requested.connect(open_settings)
    tray.quit_requested.connect(app.quit)

    # Wire listener to UI
    listener.assistant_message.connect(window.add_message)
    listener.status_changed.connect(window.set_status)
    listener.error_occurred.connect(lambda msg: window.append_log(f"[ERROR] {msg}"))
    listener.task_approval_requested.connect(window.request_task_decision)
    window.task_approved.connect(lambda task_id: listener.send_task_approval(task_id, True))
    window.task_rejected.connect(lambda task_id: listener.send_task_approval(task_id, False))

    # Wire controller to UI
    controller.engine_started.connect(lambda: window.set_engine_running(True))
    controller.engine_started.connect(lambda: tray.set_engine_running(True))
    controller.engine_started.connect(lambda: window.set_status("Bereit"))
    controller.engine_started.connect(listener.start)
    controller.engine_stopped.connect(lambda: window.set_engine_running(False))
    controller.engine_stopped.connect(lambda: tray.set_engine_running(False))
    controller.engine_stopped.connect(lambda: window.set_status("Gestoppt"))
    controller.error_occurred.connect(lambda msg: window.append_log(f"[ENGINE ERROR] {msg}"))

    # Hide window to tray if start_minimized
    if not ui_config.start_minimized:
        window.show()
    tray.show_icon()

    # Add a UI log handler that captures tarno logs
    _setup_ui_log_handler(window)

    # Start engine automatically after a short delay so the UI is visible first
    QTimer.singleShot(200, controller.start_engine)

    app.aboutToQuit.connect(controller.stop_engine)
    app.aboutToQuit.connect(listener.stop)

    sys.exit(app.exec())


def _setup_ui_log_handler(window: ControlWindow) -> None:
    """Attach a handler that streams tarno logs into the UI log view."""
    root = logging.getLogger("tarno_backend")
    handler = _UIHandler(window)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)


class _UIHandler(QObject, logging.Handler):
    """Logging handler that forwards records to the UI log view (thread-safe)."""

    log_emitted = Signal(str)

    def __init__(self, window: ControlWindow) -> None:
        QObject.__init__(self, window)
        logging.Handler.__init__(self)
        self._window = window
        self.log_emitted.connect(window.append_log)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_emitted.emit(msg)
        except Exception:
            self.handleError(record)
