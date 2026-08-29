"""Real-time audio output stream for TTS chunks."""

from __future__ import annotations

import logging
import queue
import threading
import time

import pyaudio

from tarno_backend.voice.tts_engines.base import TTSAudioChunk

log = logging.getLogger(__name__)


class TTSOutputStream:
    """Stream TTS audio chunks to the default or selected output device.

    Uses a pyaudio callback to keep latency low and to allow the synthesizer to
    keep producing chunks while audio is playing.  If a chunk arrives too late,
    silence is emitted until more data is available.
    """

    def __init__(
        self,
        sample_rate: int,
        sample_width: int = 2,
        num_channels: int = 1,
        device_index: int | None = None,
        frames_per_buffer: int = 1024,
    ) -> None:
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._num_channels = num_channels
        self._device_index = device_index
        self._frames_per_buffer = frames_per_buffer

        self._pa = pyaudio.PyAudio()
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._finished = threading.Event()
        self._started = False
        self._start_time: float | None = None
        # Gesetzt von close() - schuetzt gegen die Race Condition, bei der
        # der Timeout-Pfad in synthesizer.py._speak_streaming() den Stream
        # bereits geschlossen hat, waehrend der (daemon) Synthese-Thread noch
        # unbeaufsichtigt weiterlaeuft und versucht nachzuschreiben (siehe
        # write()/start() unten). Ohne diesen Schutz wirft
        # self._stream.start_stream() ein OSError("Stream closed"), weil
        # close() bereits self._pa.terminate() aufgerufen hat.
        self._closed = False

        try:
            self._stream = self._pa.open(
                format=self._pyaudio_format(),
                channels=self._num_channels,
                rate=self._sample_rate,
                output=True,
                output_device_index=self._device_index,
                frames_per_buffer=self._frames_per_buffer,
                stream_callback=self._callback,
            )
        except Exception:
            self._pa.terminate()
            raise

    def _pyaudio_format(self) -> int:
        """Map sample width to a pyaudio format constant."""
        if self._sample_width == 1:
            return pyaudio.paInt8
        if self._sample_width == 2:
            return pyaudio.paInt16
        if self._sample_width == 4:
            return pyaudio.paInt32
        raise ValueError(f"Unsupported sample width: {self._sample_width}")

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status: int,
    ) -> tuple[bytes, int]:
        """pyaudio output callback."""
        needed = frame_count * self._num_channels * self._sample_width

        with self._buffer_lock:
            while len(self._buffer) < needed:
                try:
                    self._buffer.extend(self._queue.get_nowait())
                except queue.Empty:
                    break

            out = bytes(self._buffer[:needed])
            self._buffer = self._buffer[needed:]

        if len(out) < needed:
            out = out.ljust(needed, b"\x00")

        if self._finished.is_set() and len(self._buffer) == 0 and self._queue.empty():
            return out, pyaudio.paComplete

        return out, pyaudio.paContinue

    def start(self) -> None:
        """Start the output stream (playback begins when chunks are written)."""
        if self._closed:
            # Stream wurde bereits geschlossen (z.B. TTS-Timeout in
            # synthesizer.py, waehrend der Synthese-Thread noch lief) -
            # still ignorieren statt gegen ein terminiertes PortAudio-Handle
            # zu laufen. Der (daemon) Aufrufer-Thread laeuft dann einfach
            # ins Leere und beendet sich, ohne den Prozess zu stoeren.
            return
        if not self._started:
            self._stream.start_stream()
            self._started = True
            self._start_time = time.monotonic()

    def write(self, chunk: TTSAudioChunk) -> None:
        """Enqueue an audio chunk for playback. No-op if already closed
        (siehe start() - schuetzt gegen die Timeout-Race-Condition)."""
        if self._closed:
            return
        if not self._started:
            self.start()
        self._queue.put(chunk.audio_bytes)

    def finish(self) -> None:
        """Signal that no more chunks will be written."""
        self._finished.set()

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Wait until all queued audio has played."""
        if not self._started:
            return True
        try:
            while self._stream.is_active():
                if self._finished.is_set() and self._queue.empty() and len(self._buffer) == 0:
                    # Give the callback one more buffer to drain.
                    time.sleep(self._frames_per_buffer / self._sample_rate)
                    break
                time.sleep(0.02)
            return True
        except Exception:
            log.exception("Fehler beim Warten auf TTS-Wiedergabe")
            return False

    def stop(self) -> None:
        """Stop playback immediately and clear buffers."""
        self._finished.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass

    def close(self) -> None:
        """Release all pyaudio resources."""
        self._closed = True
        self._finished.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                log.exception("Fehler beim Schließen des TTS-Output-Streams")
        self._pa.terminate()

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.is_active()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
