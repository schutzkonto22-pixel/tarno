"""Speech-to-text with Whisper primary, Google fallback.

Supports two listening modes:
- **Adaptive** (default when ``VADConfig`` is supplied): uses ``AdaptiveListener``
  with dynamic silence timeout and noise gate. No trailing-punctuation extension
  (that requires faster-whisper; use ``FasterWhisperRecognizer`` for full support).
- **Legacy**: falls back to ``sr.Recognizer.listen()`` when no ``VADConfig`` is
  provided or when the audio source is not an ``AudioStreamSource``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import speech_recognition as sr

if TYPE_CHECKING:
    from tarno_backend.core.config import AGCConfig, AudioConfig, VADConfig

log = logging.getLogger(__name__)


class SpeechRecognizer:
    """Captures microphone audio and transcribes it to text."""

    def __init__(
        self,
        config: AudioConfig,
        vad_config: VADConfig | None = None,
        agc_config: "AGCConfig | None" = None,
    ) -> None:
        self._config = config
        self._timeout = config.listen_timeout
        self._phrase_limit = config.phrase_time_limit
        self._energy = config.energy_threshold
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = self._energy
        self._calibrated = False

        self._has_whisper = self._check_whisper()
        if self._has_whisper:
            log.info("Spracherkennung: Whisper (lokal) + Google (Fallback)")
        else:
            log.info("Spracherkennung: Google Speech API")

        self._adaptive = None
        if vad_config is not None:
            from tarno_backend.voice.adaptive_listener import AdaptiveListener
            self._adaptive = AdaptiveListener(
                sample_rate=config.sample_rate,
                chunk_size=config.chunk_size,
                vad_config=vad_config,
                agc_config=agc_config,
            )
            log.info(
                "Adaptiver Listener aktiv (silence=%.1fs, ohne Trailing-Punctuation)",
                vad_config.silence_threshold_sec,
            )

    def calibrate(self, source: object) -> None:
        """Calibrate ambient noise from the given source once."""
        from tarno_backend.voice.audio_stream import AudioStreamSource
        if self._adaptive is not None and isinstance(source, AudioStreamSource):
            self._adaptive.calibrate(source._audio_stream)
            self._calibrated = True
            log.info("AdaptiveListener-Kalibrierung abgeschlossen")
            return

        if self._calibrated:
            return
        try:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self._calibrated = True
            log.info("Umgebungsgeräusch kalibriert (threshold=%.0f)", self._recognizer.energy_threshold)
        except Exception:
            log.exception("Kalibrierung fehlgeschlagen")

    def listen_and_recognize(self, source: object | None = None) -> str | None:
        from tarno_backend.voice.audio_stream import AudioStreamSource
        if (
            self._adaptive is not None
            and source is not None
            and isinstance(source, AudioStreamSource)
        ):
            return self._listen_adaptive(source._audio_stream)
        return self._listen_legacy(source)

    def _listen_adaptive(self, stream: object) -> str | None:
        """Adaptive path: AdaptiveListener → sr.AudioData → transcribe."""
        assert self._adaptive is not None
        pcm = self._adaptive.listen(stream)
        if pcm is None:
            return None
        audio = sr.AudioData(
            pcm.tobytes(),
            self._config.sample_rate,
            2,
        )
        return self._transcribe(audio)

    def _listen_legacy(self, source: object | None) -> str | None:
        """Legacy path: sr.Recognizer.listen() → transcribe."""
        try:
            if source is None:
                source = sr.Microphone()

            with source:
                if not self._calibrated:
                    self.calibrate(source)
                log.info("Lausche... (threshold=%.0f, timeout=%ds)", self._recognizer.energy_threshold, self._timeout)
                audio = self._recognizer.listen(
                    source,
                    timeout=self._timeout,
                    phrase_time_limit=self._phrase_limit,
                )
                log.info("Audio aufgenommen, erkenne Sprache...")
        except sr.WaitTimeoutError:
            log.info("Keine Sprache erkannt (Timeout nach %ds)", self._timeout)
            return None
        except Exception:
            log.exception("Mikrofon-Fehler")
            return None

        return self._transcribe(audio)

    def _transcribe(self, audio: sr.AudioData) -> str | None:
        language = self._config.language
        if self._has_whisper:
            try:
                text = self._recognizer.recognize_whisper(
                    audio, model="base", language=language,
                )
                log.info("Erkannt (Whisper): %s", text)
                return text.strip()
            except Exception:
                log.warning("Whisper fehlgeschlagen, versuche Google")

        try:
            # Map two-letter code to a valid Google Speech API locale.
            locale = {"de": "de-DE", "en": "en-US"}.get(language, language)
            text = self._recognizer.recognize_google(audio, language=locale)
            log.info("Erkannt (Google): %s", text)
            return text.strip()
        except sr.UnknownValueError:
            log.debug("Sprache nicht verstanden")
            return None
        except sr.RequestError:
            log.warning("Google Speech API nicht erreichbar")
            return None

    @staticmethod
    def _check_whisper() -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False
