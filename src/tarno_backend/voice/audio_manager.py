"""Central audio manager: device detection, routing decisions and state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

from tarno_backend.core.config import AudioManagerConfig

log = logging.getLogger(__name__)


class OutputDeviceType(Enum):
    HEADPHONES = auto()
    SPEAKERS = auto()
    BLUETOOTH = auto()
    UNKNOWN = auto()


class AudioState(Enum):
    NORMAL = "normal"
    TTS_ACTIVE = "tts_active"
    ECHO_RISK = "echo_risk"
    TEXT_ONLY = "text_only"
    HEADPHONE_MODE = "headphone_mode"


@dataclass
class AudioDeviceInfo:
    name: str
    device_type: OutputDeviceType
    is_default: bool


class AudioManager:
    """Detects output devices and decides whether TTS or visual output is safe."""

    def __init__(self, config: AudioManagerConfig | None = None) -> None:
        self._config = config or AudioManagerConfig()
        self._state = AudioState.NORMAL
        self._current_output = OutputDeviceType.UNKNOWN
        self._microphone_open = False
        self._headphones_connected = False

    def refresh_devices(self) -> None:
        """Re-detect the current default output device."""
        device = self._detect_default_device()
        self._current_output = device.device_type
        self._headphones_connected = device.device_type in (
            OutputDeviceType.HEADPHONES,
            OutputDeviceType.BLUETOOTH,
        )
        log.debug("Audio-Refresh: %s (%s)", device.name, device.device_type.name)

    def should_use_tts(self) -> bool:
        """Return True if TTS is unlikely to cause echo/feedback."""
        if self._state == AudioState.ECHO_RISK:
            return False
        if self._state == AudioState.TEXT_ONLY:
            return False
        if self._headphones_connected:
            return True
        # External speakers with open microphone are risky.
        if self._current_output == OutputDeviceType.SPEAKERS and self._microphone_open:
            return False
        return True

    def enter_tts(self) -> None:
        self._state = AudioState.TTS_ACTIVE

    def leave_tts(self) -> None:
        if self._headphones_connected:
            self._state = AudioState.HEADPHONE_MODE
        else:
            self._state = AudioState.NORMAL

    def set_echo_risk(self, echo_risk: bool) -> None:
        if echo_risk:
            self._state = AudioState.ECHO_RISK
        elif self._state == AudioState.ECHO_RISK:
            self._state = AudioState.NORMAL

    def set_microphone_open(self, open_: bool) -> None:
        self._microphone_open = open_

    @property
    def current_state(self) -> AudioState:
        return self._state

    @property
    def output_device(self) -> OutputDeviceType:
        return self._current_output

    def _detect_default_device(self) -> AudioDeviceInfo:
        """Platform-specific default device detection with optional pycaw."""
        import sys

        if sys.platform == "win32":
            return self._detect_windows_device()

        # Linux/macOS fallback: assume speakers until a specific adapter is added.
        return AudioDeviceInfo(name="default", device_type=OutputDeviceType.SPEAKERS, is_default=True)

    def _detect_windows_device(self) -> AudioDeviceInfo:
        try:
            from pycaw.pycaw import AudioUtilities  # type: ignore[import]
            device = AudioUtilities.GetSpeakers()
            name = device.FriendlyName
            lower = name.lower()
            if "headset" in lower or "kopfhörer" in lower or "headphone" in lower:
                device_type = OutputDeviceType.HEADPHONES
            elif "bluetooth" in lower:
                device_type = OutputDeviceType.BLUETOOTH
            else:
                device_type = OutputDeviceType.SPEAKERS
            return AudioDeviceInfo(name=name, device_type=device_type, is_default=True)
        except Exception:
            log.debug("pycaw nicht verfügbar, verwende PyAudio-Fallback")

        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            default_index = pa.get_default_output_device_info().get("index")
            info = pa.get_device_info_by_index(default_index) if default_index is not None else {}
            name = info.get("name", "unknown")
            pa.terminate()
        except Exception:
            name = "unknown"

        return AudioDeviceInfo(name=name, device_type=OutputDeviceType.UNKNOWN, is_default=True)
