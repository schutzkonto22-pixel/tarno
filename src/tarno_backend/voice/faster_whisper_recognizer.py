"""Speech-to-text using faster-whisper locally."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import speech_recognition as sr

from tarno_backend.voice.streaming_whisper_recognizer import StreamingWhisperRecognizer

if TYPE_CHECKING:
    from tarno_backend.core.config import AGCConfig, AudioConfig

log = logging.getLogger(__name__)


class FasterWhisperRecognizer:
    """Captures microphone audio and transcribes it with faster-whisper locally."""

    def __init__(self, config: AudioConfig, agc_config: AGCConfig | None = None) -> None:
        self._config = config
        self._agc_config = agc_config
        self._timeout = config.listen_timeout
        self._phrase_limit = config.phrase_time_limit
        self._energy = config.energy_threshold
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = self._energy
        self._recognizer.pause_threshold = self._config.pause_threshold
        self._calibrated = False

        try:
            from faster_whisper import WhisperModel
            cache_dir = Path(
                os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
            ) / "faster-whisper"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                config.whisper_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                download_root=str(cache_dir),
            )
            log.info(
                "Spracherkennung: faster-whisper (lokal, Modell=%s)",
                config.whisper_model,
            )
        except Exception:
            log.exception("faster-whisper konnte nicht geladen werden")
            raise

        self._streaming: StreamingWhisperRecognizer | None = None

    def calibrate(self, source: "Any") -> None:
        """Calibrate ambient noise from the given source once."""
        if self._calibrated:
            return
        try:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self._calibrated = True
            log.info("Umgebungsgeräusch kalibriert (threshold=%.0f)", self._recognizer.energy_threshold)
        except Exception:
            log.exception("Kalibrierung fehlgeschlagen")

    def listen_and_recognize(self, source: "Any" | None = None) -> str | None:
        from tarno_backend.voice.audio_stream import AudioStreamSource

        if isinstance(source, AudioStreamSource):
            if self._streaming is None:
                self._streaming = StreamingWhisperRecognizer(self._config, model=self._model, agc_config=self._agc_config)
            try:
                return self._streaming.listen_stream(source._audio_stream)
            except Exception:
                log.exception("Streaming-Spracherkennung fehlgeschlagen, falle auf energy-basierte Erkennung zurück")

        try:
            if source is None:
                source = sr.Microphone(sample_rate=16000)

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
        try:
            # faster-whisper expects a file path or numpy array
            wav_bytes = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            segments, _ = self._model.transcribe(
                tmp_path,
                language=self._config.language,
                task="transcribe",
                vad_filter=self._config.vad_filter,
                beam_size=self._config.whisper_beam_size,
                condition_on_previous_text=self._config.whisper_condition_on_previous_text,
                initial_prompt=self._config.whisper_initial_prompt or None,
            )
            text = " ".join(s.text for s in segments).strip()
            if not text:
                return None
            log.info("Erkannt (faster-whisper): %s", text)
            return text
        except Exception:
            log.exception("faster-whisper Transkription fehlgeschlagen")
            return None
