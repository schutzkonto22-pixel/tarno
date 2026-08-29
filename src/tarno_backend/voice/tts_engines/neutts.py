"""NeuTTS Nano engine with local voice cloning."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tarno_backend.voice.voice_reference import default_reference_path, default_reference_text

try:
    from neutts import NeuTTS
except ImportError:
    NeuTTS = None  # type: ignore[assignment, misc]

from tarno_backend.voice.tts_engines.base import BaseTTSEngine, TTSAudioChunk

log = logging.getLogger(__name__)

if NeuTTS is None:
    raise ImportError("neutts package is not installed")


class NeuTTSEngine(BaseTTSEngine):
    """Local voice-cloning TTS via Neuphonic NeuTTS.

    Requires ``neutts``, ``neucodec`` and, for streaming, ``llama-cpp-python``.
    Reference audio is encoded once and reused. Fallback to file-based synthesis
    is used if the GGUF streaming backend is unavailable.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._model_id = str(
            config.get("voice", "neuphonic/neutts-nano-german-q8-gguf")
        )
        self._codec_id = str(config.get("neutts_codec", "neuphonic/neucodec"))
        user_ref = config.get("voice_reference_path", "")
        user_text = config.get("voice_reference_text", "")
        self._reference_path = self._resolve_reference(user_ref)
        self._reference_text = self._resolve_reference_text(user_text)
        self._sample_rate = int(config.get("neutts_sample_rate", 24000))
        self._tts: Any | None = None

    def _load_tts(self) -> Any:
        """Lazy-load the NeuTTS model on first use."""
        if self._tts is None:
            try:
                self._tts = NeuTTS(
                    backbone_repo=self._model_id,
                    backbone_device="cpu",
                    codec_repo=self._codec_id,
                    codec_device="cpu",
                )
                log.info("NeuTTS-Engine initialisiert: %s", self._model_id)
            except Exception:
                log.exception("NeuTTS konnte nicht initialisiert werden")
                raise
        return self._tts

    @property
    def name(self) -> str:
        return "neutts"

    @property
    def sample_rate(self) -> int:
        return getattr(self, "_sample_rate", 24000)

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def supports_streaming(self) -> bool:
        # Streaming requires llama-cpp-python, a GGUF backbone and a reference.
        try:
            import llama_cpp  # noqa: F401
            return "gguf" in self._model_id.lower() and self._reference_path is not None
        except ImportError:
            return False

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    def _resolve_reference(self, path: Any) -> Path | None:
        if path:
            p = Path(str(path)).expanduser()
            if p.is_absolute() and p.exists():
                return p
            # Try relative to package root (source/PyInstaller builds).
            root = Path(__file__).resolve().parent.parent.parent
            rel = root / p
            if rel.exists():
                return rel
            log.warning("NeuTTS Referenzsample nicht gefunden: %s", p)

        # Use the bundled Thorsten-Voice reference as a sensible default.
        default = default_reference_path()
        if default.exists():
            log.info("Verwende Standard-Referenzsample: %s", default)
            return default
        return None

    def _resolve_reference_text(self, text: Any) -> str:
        if text:
            return str(text)
        return default_reference_text()

    def _ensure_reference(self, text: str) -> tuple[Any, str]:
        """Encode the reference audio once and return (codes, text)."""
        ref_path = self._reference_path
        ref_text = self._reference_text

        if ref_path is None:
            log.warning("Kein Voice-Cloning-Referenzsample konfiguriert; verwende Default-Stimme.")
            return None, ""

        if not ref_text:
            ref_text = text

        try:
            tts = self._load_tts()
            codes = tts.encode_reference(str(ref_path))
            return codes, ref_text
        except Exception:
            log.exception("NeuTTS Referenz-Encodierung fehlgeschlagen: %s", ref_path)
            return None, ""

    def synthesize_stream(self, text: str) -> Iterator[TTSAudioChunk]:
        """Stream audio chunks using NeuTTS.infer_stream."""
        codes, ref_text = self._ensure_reference(text)

        try:
            tts = self._load_tts()
            for chunk in tts.infer_stream(text, codes, ref_text):
                # NeuTTS returns float32 [-1, 1]; convert to int16 PCM.
                audio = (np.asarray(chunk) * 32767).astype(np.int16)
                yield TTSAudioChunk(
                    audio_bytes=audio.tobytes(),
                    sample_rate=self.sample_rate,
                    sample_width=self.sample_width,
                    num_channels=self.num_channels,
                )
        except Exception:
            log.exception("NeuTTS Streaming fehlgeschlagen; falle auf Dateisynthese zurück")
            # If streaming fails, synthesize to a temp WAV and yield one chunk.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = Path(tmp.name)
            if self.synthesize_to_file(text, output_path):
                data = output_path.read_bytes()
                yield TTSAudioChunk(
                    audio_bytes=data,
                    sample_rate=self.sample_rate,
                    sample_width=self.sample_width,
                    num_channels=self.num_channels,
                )

    def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Synthesize the whole utterance to a WAV file."""
        codes, ref_text = self._ensure_reference(text)

        try:
            tts = self._load_tts()
            wav = tts.infer(text, codes, ref_text)
            import soundfile as sf

            sf.write(str(output_path), wav, self.sample_rate)
            return True
        except Exception:
            log.exception("NeuTTS Dateisynthese fehlgeschlagen")
            return False
