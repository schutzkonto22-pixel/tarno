"""Background thread for wake word detection and speech recognition."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from tarno_backend.voice.audio_stream import AudioStream, AudioStreamSource

if TYPE_CHECKING:
    from tarno_backend.core.config import TarnoConfig
    from tarno_backend.voice.recognizer import SpeechRecognizer

log = logging.getLogger(__name__)


class VoiceWorker(QObject):
    """Runs wake word + STT on a dedicated thread."""

    wake_word_detected = Signal()
    play_confirmation = Signal()
    speech_recognized = Signal(str)
    listening_started = Signal()
    listening_stopped = Signal()
    error_occurred = Signal(str)

    # How long to wait for the LLM response + TTS playback to finish before
    # giving up on the conversation window and falling back to wake-word mode.
    # Generous on purpose: this only starts ticking once we're waiting for
    # signal_turn_ready() (i.e. after TARNO has *finished* speaking, never
    # while TTS is still playing), so it just needs to outlast slow LLM
    # calls + long spoken responses, not any single turn's actual duration.
    _TURN_TIMEOUT = 45.0

    def __init__(self, config: "TarnoConfig") -> None:
        super().__init__()
        self._config = config
        self._active = False
        self._running = False
        self._in_conversation = False
        self._turn_ready = threading.Event()

    @Slot()
    def start_listening(self) -> None:
        if self._active:
            return
        self._active = True
        self._running = True
        self._run_loop()

    @Slot()
    def stop_listening(self) -> None:
        self._active = False
        self._running = False
        self._in_conversation = False
        self._turn_ready.set()

    def signal_turn_ready(self) -> None:
        """Called (from the mediator, on the Qt main thread) once TARNO has
        finished speaking its response — tells the voice thread it may keep
        listening for a follow-up instead of returning to wake-word mode."""
        self._turn_ready.set()

    def _run_loop(self) -> None:
        from tarno_backend.voice.recognizer import SpeechRecognizer
        from tarno_backend.voice.wakeword import WakeWordDetector

        recognizer = SpeechRecognizer(
            self._config.audio, vad_config=self._config.vad, agc_config=self._config.agc
        )

        if not self._config.wakeword.enabled:
            self._run_push_to_talk(recognizer)
            return

        try:
            wakeword = WakeWordDetector(self._config.wakeword, agc_config=self._config.agc)
        except Exception as exc:
            log.exception("Wake-Word init fehlgeschlagen")
            self.error_occurred.emit(f"Wake-Word Fehler: {exc}")
            self._run_push_to_talk(recognizer)
            return

        try:
            with AudioStream(self._config.audio) as stream:
                log.info("Wake-Word-Erkennung aktiv")
                self.listening_started.emit()
                recognizer.calibrate(AudioStreamSource(stream))

                while self._active:
                    chunk = stream.read_chunk()
                    if not wakeword.process_frame(chunk):
                        continue

                    wakeword.reset()
                    self.wake_word_detected.emit()

                    if self._config.wakeword.confirm_wake_word:
                        self.play_confirmation.emit()
                    self._turn_ready.clear()
                    heard_something = self._recognize_and_emit(recognizer, stream)

                    if heard_something and self._turn_ready.wait(timeout=self._TURN_TIMEOUT):
                        self._in_conversation = True
                        self._conversation_loop(recognizer, stream)
                        self._in_conversation = False

                    if self._active:
                        stream.flush_input_buffer()
                        wakeword.reset()

        except Exception as exc:
            log.exception("Voice-Loop Fehler")
            self.error_occurred.emit(str(exc))
        finally:
            self.listening_stopped.emit()

    def _run_push_to_talk(self, recognizer: "SpeechRecognizer") -> None:
        self.listening_started.emit()
        while self._active:
            self._recognize_and_emit(recognizer)
        self.listening_stopped.emit()

    def _conversation_loop(
        self,
        recognizer: "SpeechRecognizer",
        stream: "AudioStream",
    ) -> None:
        while self._active:
            text = recognizer.listen_and_recognize(AudioStreamSource(stream))
            if not text:
                log.info("Kein Follow-up erkannt — zurück zum Wake-Word-Modus")
                return
            self._turn_ready.clear()
            self.speech_recognized.emit(text)
            if not self._turn_ready.wait(timeout=self._TURN_TIMEOUT):
                log.info("Keine Antwort rechtzeitig gesprochen — zurück zum Wake-Word-Modus")
                return

    def _recognize_and_emit(
        self,
        recognizer: "SpeechRecognizer",
        stream: "AudioStream" | None = None,
    ) -> bool:
        source = AudioStreamSource(stream) if stream else None
        text = recognizer.listen_and_recognize(source)
        if not text:
            return False
        self.speech_recognized.emit(text)
        return True
