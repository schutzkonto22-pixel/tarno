"""Framework-agnostic voice pipeline with formal state machine and watchdog.

Provides a robust voice loop: Wake-Word → STT → LLM → TTS → Wake-Word.
The AudioStream is kept alive across interactions (pause/resume), preventing
the audio-device-handle loss that caused wake-word failures after the first
interaction.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from tarno_backend.voice.echo_protection import EchoProtection

if TYPE_CHECKING:
    from tarno_backend.core.config import TarnoConfig

log = logging.getLogger(__name__)


class VoiceState(Enum):
    """Formal voice pipeline states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


# Legacy string constants for backward compatibility with gRPC server callbacks.
STATE_IDLE = VoiceState.IDLE.value
STATE_LISTENING = VoiceState.LISTENING.value
STATE_PROCESSING = VoiceState.PROCESSING.value


class VoiceController:
    """Runs wake word detection + speech recognition on a background thread.

    Key design decisions (Phase 2 stabilisation):
    - AudioStream uses pause()/resume() instead of stop()/start() so the
      PyAudio instance stays alive across interactions.
    - A watchdog thread monitors the voice loop and restarts it if it hangs.
    - All exceptions inside the loop are caught, logged and recovered from.
    """

    _TURN_TIMEOUT = 45.0
    _WATCHDOG_INTERVAL = 10.0  # seconds between watchdog heartbeat checks
    _WATCHDOG_TIMEOUT = 120.0  # max seconds without heartbeat before restart

    def __init__(
        self,
        config: "TarnoConfig",
        on_wake_word: Callable[[], None] | None = None,
        on_speech: Callable[[str], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._on_wake_word = on_wake_word or (lambda: None)
        self._on_speech = on_speech or (lambda text: None)
        self._on_state = on_state or (lambda state: None)
        self._on_error = on_error or (lambda msg: None)

        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._active = False
        self._turn_ready = threading.Event()
        self._state = VoiceState.IDLE
        self._last_heartbeat = 0.0
        self._restart_count = 0
        self._interaction_count = 0

        # Settings-triggered Vosk small/large model hot-swap: written from
        # whatever thread handles the gRPC "SetWakeWordModelSize" request,
        # read/consumed once per wake-word poll iteration on the voice
        # thread - a lock avoids torn reads even though CPython attribute
        # assignment is already atomic, since this also guards
        # self._wakeword_detector across threads.
        self._model_swap_lock = threading.Lock()
        self._pending_vosk_model_size: str | None = None
        self._pending_vosk_model_callback: Callable[[bool, str], None] | None = None
        self._wakeword_detector = None

        # Settings-triggered microphone-device hot-swap - same pattern as the
        # Vosk model-size swap above, applied on the voice thread between
        # wake-word poll iterations. AudioStream has no live "change device"
        # method, so applying this means recreating the stream instance.
        self._mic_swap_lock = threading.Lock()
        self._pending_microphone_device: str | None = None
        self._pending_microphone_callback: Callable[[bool, str], None] | None = None

        # Suppresses wake-word processing while TARNO is speaking (+ a short
        # cooldown after) - found live: this was only ever wired into the
        # older console-mode loop (tarno/core/engine.py), never into this
        # gRPC/WinUI-used controller. Without it, the wake-word scanner kept
        # listening through TTS playback regardless of what triggered that
        # TTS (voice OR a typed chat message), so TARNO could hear its own
        # voice, self-trigger the wake word on itself, and end up talking to
        # itself in a loop. on_tts_started/on_tts_finished are called from
        # the gRPC bridge's TtsStartedEvent/TtsFinishedEvent subscribers,
        # which fire for ANY TTS playback regardless of trigger source.
        self._echo_protection = EchoProtection()

    def on_tts_started(self) -> None:
        self._echo_protection.on_tts_started()

    def on_tts_finished(self) -> None:
        self._echo_protection.on_tts_finished()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._turn_ready.clear()
        self._last_heartbeat = time.monotonic()
        self._restart_count = 0
        self._thread = threading.Thread(
            target=self._run_loop_safe, daemon=True, name="VoiceController"
        )
        self._thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, daemon=True, name="VoiceWatchdog"
        )
        self._watchdog_thread.start()
        log.info("VoiceController gestartet (mit Watchdog)")

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._turn_ready.set()  # unblock any waits
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None
        self._watchdog_thread = None
        self._set_state(VoiceState.IDLE)
        log.info(
            "VoiceController gestoppt (Interaktionen: %d, Neustarts: %d)",
            self._interaction_count,
            self._restart_count,
        )

    def signal_turn_ready(self) -> None:
        """Call once TARNO has finished speaking — unblocks the conversation window."""
        self._turn_ready.set()

    def request_vosk_model_size(
        self, model_size: str, on_result: Callable[[bool, str], None] | None = None
    ) -> None:
        """Settings small/large toggle: queues a hot-swap, applied on the
        voice thread the next time it's between wake-word poll iterations
        (not blocking the caller - the actual download/load can take a
        while for an uncached large model). on_result(success, model_size)
        fires from the voice thread once the swap attempt finishes, if given -
        callers on another thread (e.g. the gRPC event loop) must hop back
        themselves; on_result implementations here just call self._broadcast,
        which is already safe to invoke from a foreign thread."""
        with self._model_swap_lock:
            self._pending_vosk_model_size = model_size
            self._pending_vosk_model_callback = on_result

    @property
    def vosk_model_size(self) -> str | None:
        detector = self._wakeword_detector
        return getattr(detector, "_vosk_model_size", None) if detector is not None else None

    def request_microphone_device(
        self, device: str, on_result: Callable[[bool, str], None] | None = None
    ) -> None:
        """Settings microphone picker: queues a hot-swap, applied on the
        voice thread between wake-word poll iterations (same shape as
        request_vosk_model_size). on_result(success, device) fires from the
        voice thread once the swap attempt finishes."""
        with self._mic_swap_lock:
            self._pending_microphone_device = device
            self._pending_microphone_callback = on_result

    # ── State Machine ──────────────────────────────────────────────

    def _set_state(self, new_state: VoiceState) -> None:
        old = self._state
        self._state = new_state
        if old != new_state:
            log.debug("Voice-State: %s → %s", old.value, new_state.value)
        self._on_state(new_state.value)

    # ── Watchdog ───────────────────────────────────────────────────

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def _watchdog(self) -> None:
        """Monitor the voice thread and restart it if it hangs."""
        log.debug("Voice-Watchdog gestartet")
        while self._active:
            time.sleep(self._WATCHDOG_INTERVAL)
            if not self._active:
                break
            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self._WATCHDOG_TIMEOUT:
                log.warning(
                    "Voice-Watchdog: kein Heartbeat seit %.0fs — starte Voice-Loop neu",
                    elapsed,
                )
                self._on_error(f"Voice-Pipeline hängt seit {elapsed:.0f}s — automatischer Neustart")
                self._restart_voice_loop()
        log.debug("Voice-Watchdog beendet")

    def _restart_voice_loop(self) -> None:
        """Kill the current voice thread and start a new one."""
        self._restart_count += 1
        old_thread = self._thread
        # Signal the old loop to exit
        self._turn_ready.set()
        if old_thread is not None and old_thread is not threading.current_thread():
            old_thread.join(timeout=3)
        self._last_heartbeat = time.monotonic()
        self._thread = threading.Thread(
            target=self._run_loop_safe, daemon=True, name="VoiceController"
        )
        self._thread.start()
        log.info("Voice-Loop neu gestartet (Neustart #%d)", self._restart_count)

    # ── Main Loop ──────────────────────────────────────────────────

    def _run_loop_safe(self) -> None:
        """Outer wrapper that catches all exceptions and attempts recovery."""
        while self._active:
            try:
                self._run_loop()
            except Exception as exc:
                log.exception("VoiceController-Loop Fehler — versuche Neustart")
                self._on_error(f"Voice-Fehler: {exc}")
                self._set_state(VoiceState.ERROR)
                if not self._active:
                    break
                time.sleep(2.0)
                self._restart_count += 1
                log.info("Voice-Loop Recovery (Versuch #%d)", self._restart_count)
            else:
                # Loop exited cleanly (self._active was set to False)
                break
        self._set_state(VoiceState.IDLE)

    def _run_loop(self) -> None:
        from tarno_backend.voice.audio_stream import AudioStream, AudioStreamSource
        from tarno_backend.voice.faster_whisper_recognizer import FasterWhisperRecognizer
        from tarno_backend.voice.wakeword import WakeWordDetector

        # FasterWhisperRecognizer (not the legacy SpeechRecognizer) is required
        # here: only it wires partial_transcribe into AdaptiveListener, which
        # is what lets sentence_seems_incomplete() actually grant a trailing
        # extension when the user pauses mid-sentence to think. With
        # SpeechRecognizer, AdaptiveListener._try_extend() always returns
        # False immediately (no partial_transcribe callback), so any pause
        # over silence_threshold_sec silently cuts the utterance off - found
        # live: "waits politely while you finish speaking" never engaged.
        recognizer = FasterWhisperRecognizer(self._config.audio, agc_config=self._config.agc)

        try:
            wakeword = WakeWordDetector(self._config.wakeword, agc_config=self._config.agc)
        except Exception as exc:
            log.exception("Wake-Word init fehlgeschlagen")
            self._on_error(f"Wake-Word Fehler: {exc}")
            self._active = False
            return
        self._wakeword_detector = wakeword

        stream = AudioStream(self._config.audio)
        try:
            stream.start()
            log.info("Wake-Word-Erkennung aktiv (VoiceController)")
            source = AudioStreamSource(stream)
            recognizer.calibrate(source)
            self._set_state(VoiceState.IDLE)

            while self._active:
                self._heartbeat()

                # ── Apply a pending Settings small/large model swap, if any ──
                with self._model_swap_lock:
                    pending_size = self._pending_vosk_model_size
                    pending_callback = self._pending_vosk_model_callback
                    self._pending_vosk_model_size = None
                    self._pending_vosk_model_callback = None
                if pending_size is not None:
                    try:
                        wakeword.set_vosk_model_size(pending_size)
                        if pending_callback:
                            pending_callback(True, pending_size)
                    except Exception:
                        log.exception("Vosk-Modell-Wechsel zu '%s' fehlgeschlagen", pending_size)
                        self._on_error(f"Vosk-Modell-Wechsel fehlgeschlagen: {pending_size}")
                        if pending_callback:
                            pending_callback(False, pending_size)

                # ── Apply a pending Settings microphone-device swap, if any ──
                with self._mic_swap_lock:
                    pending_device = self._pending_microphone_device
                    pending_mic_callback = self._pending_microphone_callback
                    self._pending_microphone_device = None
                    self._pending_microphone_callback = None
                if pending_device is not None:
                    try:
                        stream.stop()
                        self._config.audio.microphone_device = pending_device
                        stream = AudioStream(self._config.audio)
                        stream.start()
                        source = AudioStreamSource(stream)
                        wakeword.reset()
                        log.info("Mikrofon gewechselt zu: %s", pending_device)
                        if pending_mic_callback:
                            pending_mic_callback(True, pending_device)
                    except Exception:
                        log.exception("Mikrofon-Wechsel zu '%s' fehlgeschlagen", pending_device)
                        self._on_error(f"Mikrofon-Wechsel fehlgeschlagen: {pending_device}")
                        if pending_mic_callback:
                            pending_mic_callback(False, pending_device)

                # ── Read audio chunk for wake-word ──
                try:
                    chunk = stream.read_chunk()
                except RuntimeError:
                    log.warning("AudioStream Lesefehler — versuche Neustart")
                    if stream.restart():
                        source = AudioStreamSource(stream)
                        wakeword.reset()
                        continue
                    raise

                if self._echo_protection.is_suppressed():
                    # TARNO is currently speaking (or just finished, within
                    # the cooldown window) - skip wake-word processing so it
                    # cannot hear and re-trigger on its own TTS output.
                    continue

                if not wakeword.process_frame(chunk):
                    continue

                # ── Wake word detected ──
                wakeword.reset()
                self._on_wake_word()
                self._interaction_count += 1
                log.info(
                    "Wake-Word erkannt (Interaktion #%d)",
                    self._interaction_count,
                )

                # Keep the raw stream running (not paused) for STT: recognition
                # must reuse this same live AudioStream via AudioStreamSource so
                # the AdaptiveListener/VAD path (whisper-sensitive silence
                # detection, configured microphone_device) actually engages -
                # previously the stream was paused here and STT silently fell
                # back to a brand-new default sr.Microphone() with a rigid
                # legacy timeout, bypassing the VAD config entirely.
                self._set_state(VoiceState.LISTENING)
                self._turn_ready.clear()

                heard_something = self._recognize_and_emit(recognizer, source)

                if heard_something:
                    self._set_state(VoiceState.PROCESSING)
                    # Pause while TARNO thinks/speaks so the mic doesn't buffer
                    # its own TTS output (or dead air) for the next listen.
                    stream.pause()
                    turn_completed = self._turn_ready.wait(timeout=self._TURN_TIMEOUT)
                    if self._active:
                        try:
                            stream.resume()
                        except Exception:
                            log.warning("Stream-Resume fehlgeschlagen — versuche Neustart")
                            if not stream.restart():
                                raise
                        source = AudioStreamSource(stream)
                        stream.flush_input_buffer()
                        if turn_completed:
                            self._conversation_loop(recognizer, stream, source)

                # ── Resume wake-word listening ──
                if self._active:
                    wakeword.reset()
                    self._set_state(VoiceState.IDLE)

        finally:
            self._wakeword_detector = None
            stream.stop()

    def _conversation_loop(self, recognizer: "Any", stream: "Any", source: "Any") -> None:
        """Keep listening for follow-up commands until silence timeout."""
        from tarno_backend.voice.audio_stream import AudioStreamSource

        while self._active:
            self._heartbeat()
            self._set_state(VoiceState.LISTENING)
            text = recognizer.listen_and_recognize(source)
            if not text:
                log.info("Kein Follow-up erkannt — zurück zum Wake-Word-Modus")
                return
            self._interaction_count += 1
            self._turn_ready.clear()
            self._on_speech(text)
            self._set_state(VoiceState.PROCESSING)
            stream.pause()
            turn_completed = self._turn_ready.wait(timeout=self._TURN_TIMEOUT)
            if not self._active:
                return
            try:
                stream.resume()
            except Exception:
                log.warning("Stream-Resume fehlgeschlagen — versuche Neustart")
                if not stream.restart():
                    raise
            source = AudioStreamSource(stream)
            stream.flush_input_buffer()
            if not turn_completed:
                log.info("Keine Antwort rechtzeitig gesprochen — zurück zum Wake-Word-Modus")
                return

    def _recognize_and_emit(self, recognizer: "Any", source: "Any") -> bool:
        text = recognizer.listen_and_recognize(source)
        if not text:
            return False
        self._on_speech(text)
        return True
