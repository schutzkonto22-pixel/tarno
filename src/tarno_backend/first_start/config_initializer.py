"""Write the user configuration collected by the first-start wizard."""

from __future__ import annotations

import logging
from pathlib import Path

from tarno_backend.core.config import TarnoConfig

log = logging.getLogger(__name__)


def default_user_config_path() -> Path:
    return Path.home() / ".tarno" / "config" / "tarno_config.yaml"


class ConfigInitializer:
    """Collects wizard choices and persists a valid TARNO user config."""

    def __init__(self, config: TarnoConfig | None = None) -> None:
        self.config = config or TarnoConfig.load()

    def set_provider(self, provider: str, api_key: str) -> None:
        self.config.llm.provider = provider
        # The API key is intentionally NOT stored in the config file.
        # TARNO reads it from the MISTRAL_API_KEY environment variable.
        # The api_key argument is kept for API compatibility but ignored.

    def set_privacy(self, pii_redaction: bool, crash_reporting: bool) -> None:
        self.config.security.pii_redaction_enabled = pii_redaction
        self.config.telemetry.crash_reporting = crash_reporting

    def set_audio(self, input_device: int | None, whisper_model: str) -> None:
        # Audio device selection is stored in a separate runtime profile;
        # the core AudioConfig currently exposes only whisper_model here.
        self.config.audio.whisper_model = whisper_model

    def set_wakeword(
        self, enabled: bool, model_name: str, backend: str = "openwakeword"
    ) -> None:
        self.config.wakeword.enabled = enabled
        self.config.wakeword.model_name = model_name
        self.config.wakeword.backend = backend

    def save(self, path: Path | None = None) -> Path:
        target = path or default_user_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.config.save(target)
        log.info("Benutzerkonfiguration gespeichert: %s", target)
        return target


def required_models_for_config(config: TarnoConfig) -> list[dict]:
    """Return the list of models that should be downloaded for this profile.

    Wake-word models are bundled with the application, so they are not listed
    here. faster-whisper models are downloaded automatically by the
    faster-whisper library on first use, so they are also not listed here.
    """
    return []
