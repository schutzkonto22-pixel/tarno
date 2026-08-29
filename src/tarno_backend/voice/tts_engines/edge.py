"""Network TTS fallback using edge-tts and gTTS."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tarno_backend.voice.tts_engines.base import BaseTTSEngine, TTSAudioChunk

log = logging.getLogger(__name__)


class EdgeTTSEngine(BaseTTSEngine):
    """edge-tts / gTTS fallback. Produces an MP3 file, not a PCM stream."""

    _VOICE_BY_LANGUAGE = {
        "de": "de-DE-ConradNeural",
        "en": "en-US-GuyNeural",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._language = str(config.get("language", "de"))
        # The configured voice may be a HuggingFace repo id (Piper), which edge-tts
        # cannot use. Derive a valid Microsoft neural voice from the language.
        configured_voice = str(config.get("voice", ""))
        if configured_voice and "-" in configured_voice and "Neural" in configured_voice:
            self._voice = configured_voice
        else:
            self._voice = self._VOICE_BY_LANGUAGE.get(self._language, "de-DE-ConradNeural")
        self._speed = float(config.get("speed", 1.1))
        self._rate = f"{int((self._speed - 1.0) * 100):+d}%"
        self._has_edge = self._check_edge()

    @property
    def name(self) -> str:
        return "edge-tts"

    @property
    def sample_rate(self) -> int:
        # pygame will resample/playback the MP3; this value is mostly symbolic.
        return 24000

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def supports_streaming(self) -> bool:
        return False

    def synthesize_stream(self, text: str) -> Iterator[TTSAudioChunk]:
        """edge-tts cannot stream raw PCM; the caller should use synthesize_to_file."""
        raise NotImplementedError("Edge engine only supports file output")

    def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Save TTS to an MP3 using edge-tts or gTTS."""
        if self._has_edge:
            try:
                return self._synthesize_edge(text, output_path)
            except Exception:
                log.warning("edge-tts fehlgeschlagen, versuche gTTS", exc_info=True)
        return self._synthesize_gtts(text, output_path)

    def _synthesize_edge(self, text: str, output_path: Path) -> bool:
        import edge_tts

        async def _run() -> None:
            communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
            await communicate.save(str(output_path))

        try:
            asyncio.run(_run())
        except Exception:
            log.exception("edge-tts Synthese fehlgeschlagen")
            return False

        return output_path.exists() and output_path.stat().st_size > 0

    def _synthesize_gtts(self, text: str, output_path: Path) -> bool:
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self._language, slow=False)
            tts.save(str(output_path))
        except Exception:
            log.exception("gTTS Synthese fehlgeschlagen")
            return False

        return output_path.exists() and output_path.stat().st_size > 0

    @staticmethod
    def _check_edge() -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False
