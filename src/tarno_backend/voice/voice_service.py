"""TARNO voice service that connects microphone input to the OpenVoiceOS bus."""

from __future__ import annotations

import logging
import threading
import time

from ovos_bus_client import MessageBusClient
from ovos_bus_client.message import Message

from tarno_backend.core.config import TarnoConfig
from tarno_backend.telemetry.logging import new_correlation_id, set_correlation_id
from tarno_backend.voice.audio_stream import AudioStream, AudioStreamSource
from tarno_backend.voice.faster_whisper_recognizer import FasterWhisperRecognizer
from tarno_backend.voice.synthesizer import SpeechSynthesizer
from tarno_backend.voice.wakeword import WakeWordDetector

log = logging.getLogger(__name__)


class VoiceService:
    """Listen for wake word, transcribe speech, and speak responses via the OVOS bus."""

    def __init__(self, config: TarnoConfig, bus: MessageBusClient) -> None:
        set_correlation_id(new_correlation_id())
        self._config = config
        self._bus = bus
        self._audio = AudioStream(config.audio)
        self._wakeword = WakeWordDetector(config.wakeword, agc_config=config.agc)
        self._recognizer = FasterWhisperRecognizer(config.audio, agc_config=config.agc)
        self._synthesizer = SpeechSynthesizer(config.audio)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the voice pipeline and subscribe to bus events."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                log.warning("TARNO Voice Service ist bereits aktiv")
                return
            self._bus.on("speak", self._handle_speak)
            self._running = True
            self._audio.start()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        log.info("Starte TARNO Voice Service")

    def stop(self) -> None:
        """Stop the voice pipeline."""
        with self._lock:
            self._running = False
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._audio.stop()
        self._synthesizer.stop()
        log.info("TARNO Voice Service gestoppt")

    def _handle_speak(self, message: Message) -> None:
        """Play TTS for 'speak' messages from the bus."""
        text = message.data.get("utterance", "")
        if text:
            self._emit_status("Spricht...")
            self._synthesizer.speak(text)
            self._emit_status("Bereit")

    def _emit_status(self, state: str) -> None:
        """Emit a status message on the bus for the UI."""
        self._bus.emit(Message("tarno.status", data={"state": state}))

    def _run_loop(self) -> None:
        """Continuously listen for the wake word and process speech."""
        try:
            self._recognizer.calibrate(AudioStreamSource(self._audio))
            self._emit_status("Bereit")
            while self._running:
                try:
                    chunk = self._audio.read_chunk()
                except Exception as exc:
                    log.warning("Audio-Lesefehler: %s", exc)
                    time.sleep(0.2)
                    continue

                if not self._wakeword.process_frame(chunk):
                    continue

                log.info("Wake-Word erkannt, aktiviere Spracherkennung")
                self._emit_status("Hört zu...")
                if self._config.wakeword.confirm_wake_word:
                    self._synthesizer.play_sound()
                text = self._recognizer.listen_and_recognize(AudioStreamSource(self._audio))
                self._audio.flush_input_buffer()
                if not self._running:
                    break
                if text:
                    self._send_utterance(text)
                if self._running:
                    self._emit_status("Bereit")
        except Exception:
            log.exception("Voice-Loop Fehler")
            self._emit_status("Fehler")

    def _send_utterance(self, text: str) -> None:
        """Send the transcribed utterance to the OVOS bus."""
        log.info("Sende Utterance an Bus: %s", text)
        message = Message(
            "recognizer_loop:utterance",
            data={"utterances": [text], "lang": self._config.language},
        )
        self._bus.emit(message)
