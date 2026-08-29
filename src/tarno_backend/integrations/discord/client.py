"""Local Discord client microphone control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DiscordClientConfig:
    enabled: bool = False
    ptt_key: str | None = None
    virtual_cable: str | None = None


class DiscordClientModule:
    """Controls the local Discord client by simulating the configured PTT key.

    A future extension can route audio through a virtual cable configured via
    `virtual_cable`.
    """

    def __init__(self, config: DiscordClientConfig | None = None) -> None:
        self._config = config or DiscordClientConfig()
        self._ptt_active = False

    def set_mute(self, muted: bool) -> str:
        if not self._config.enabled:
            return "Discord-Integration ist deaktiviert."
        # Muting via PTT simulation: release PTT when muted.
        if muted and self._ptt_active:
            self.push_to_talk(False)
        return f"Discord Mikrofon {'stumm' if muted else 'aktiv'} (simuliert)."

    def push_to_talk(self, active: bool) -> str:
        if not self._config.enabled:
            return "Discord-Integration ist deaktiviert."
        if not self._config.ptt_key:
            return "Keine PTT-Taste in der Konfiguration hinterlegt."

        try:
            from pynput.keyboard import Controller, Key
        except ImportError:
            return "pynput ist nicht installiert; PTT-Steuerung nicht verfügbar."

        keyboard = Controller()
        key = self._resolve_key(Key, self._config.ptt_key)
        if key is None:
            return f"Unbekannte Taste: {self._config.ptt_key}"

        if active:
            keyboard.press(key)
        else:
            keyboard.release(key)
        self._ptt_active = active
        return f"Discord PTT {'gedrückt' if active else 'losgelassen'}."

    @staticmethod
    def _resolve_key(key_module: Any, name: str) -> Any:
        try:
            return getattr(key_module, name)
        except AttributeError:
            return name if len(name) == 1 else None
