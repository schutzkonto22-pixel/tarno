"""Silero VAD wrapper for streaming voice activity detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

log = logging.getLogger(__name__)


class SileroVAD:
    """Stream VAD using the Silero ONNX model.

    The model works at 16 kHz mono.  Chunks are expected to be a multiple of
    512 samples (the native VAD window).  ``process_chunk`` returns a start/end
    timestamp in samples when speech begins/ends.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._sampling_rate = int(cfg.get("sampling_rate", 16000))
        self._threshold = float(cfg.get("start_threshold", 0.5))
        self._min_silence_duration_ms = int(cfg.get("min_silence_duration_ms", 500))
        self._speech_pad_ms = int(cfg.get("speech_pad_ms", 30))
        self._chunk_size = int(cfg.get("chunk_size", 512))

        try:
            self._model = load_silero_vad()
        except Exception:
            log.exception("Silero VAD Modell konnte nicht geladen werden")
            raise

        self._iterator = VADIterator(
            self._model,
            threshold=self._threshold,
            sampling_rate=self._sampling_rate,
            min_silence_duration_ms=self._min_silence_duration_ms,
            speech_pad_ms=self._speech_pad_ms,
        )

    def reset(self) -> None:
        """Reset internal VAD state for a new utterance."""
        self._iterator.reset_states()

    def process_chunk(self, pcm_bytes: bytes) -> dict[str, int] | None:
        """Process one audio chunk.

        Args:
            pcm_bytes: little-endian 16-bit mono PCM.

        Returns:
            ``{"start": samples}`` when speech starts, ``{"end": samples}``
            when it ends, or ``None`` if the state did not change.
        """
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Pad or trim to the expected chunk size to avoid shape issues.
            expected = self._chunk_size
            if len(samples) < expected:
                samples = np.pad(samples, (0, expected - len(samples)), mode="constant")
            elif len(samples) > expected:
                samples = samples[:expected]

            tensor = torch.from_numpy(samples)
            return self._iterator(tensor, return_seconds=False)
        except Exception:
            log.exception("Silero VAD Verarbeitungsfehler")
            return None

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Return whether the current chunk is classified as speech."""
        result = self.process_chunk(pcm_bytes)
        if result is None:
            return self._iterator.triggered
        # If a start marker arrives, we are in speech; end marker means not.
        if "start" in result:
            return True
        if "end" in result:
            return False
        return self._iterator.triggered
