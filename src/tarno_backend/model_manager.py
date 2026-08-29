"""Manage local AI/ML models for TARNO.

Handles model discovery, download, SHA-256 verification and resume support.
Models are stored under ``%LOCALAPPDATA%/.tarno/models`` by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Description of a downloadable model."""

    name: str
    filename: str
    url: str
    sha256: str | None = None
    size: int | None = None


@dataclass
class DownloadProgress:
    """Progress snapshot passed to callbacks."""

    model: str
    downloaded: int
    total: int | None = None


def _default_models_dir() -> Path:
    return Path.home() / ".tarno" / "models"


class ModelManager:
    """Download and verify the models required by TARNO."""

    def __init__(
        self,
        models_dir: Path | str | None = None,
        models_config: list[dict[str, Any]] | None = None,
    ) -> None:
        self._models_dir = Path(models_dir).expanduser() if models_dir else _default_models_dir()
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._config = models_config or []

    @staticmethod
    def _sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _model_path(self, model: ModelInfo) -> Path:
        return self._models_dir / model.filename

    def required_models(self) -> list[ModelInfo]:
        """Return the list of configured models."""
        models: list[ModelInfo] = []
        for entry in self._config:
            try:
                models.append(ModelInfo(**entry))
            except TypeError:
                log.warning("Ungültiger Modelleintrag: %s", entry)
        return models

    def is_downloaded(self, model: ModelInfo) -> bool:
        path = self._model_path(model)
        if not path.exists():
            return False
        if model.sha256:
            return self._sha256_of(path) == model.sha256.lower()
        return path.stat().st_size > 0

    def download(
        self,
        model: ModelInfo,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
        chunk_size: int = 8192,
    ) -> bool:
        """Download a single model, optionally resuming an interrupted transfer."""
        path = self._model_path(model)
        existing = path.stat().st_size if path.exists() else 0

        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            log.info("Setze Download für %s bei Byte %s fort", model.name, existing)

        try:
            with requests.get(model.url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = existing + int(response.headers.get("content-length", 0) or 0)
                if model.size and total < model.size:
                    total = model.size

                mode = "ab" if existing and response.status_code == 206 else "wb"
                downloaded = existing if mode == "ab" else 0
                with path.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(DownloadProgress(
                                model=model.name,
                                downloaded=downloaded,
                                total=total,
                            ))
        except Exception:
            log.exception("Download für %s fehlgeschlagen", model.name)
            return False

        if model.sha256 and self._sha256_of(path) != model.sha256.lower():
            log.error("SHA256-Prüfung für %s fehlgeschlagen", model.name)
            path.unlink(missing_ok=True)
            return False

        log.info("Modell %s erfolgreich heruntergeladen", model.name)
        return True

    def download_all(
        self,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> dict[str, bool]:
        """Download all required models and return per-model success state."""
        results: dict[str, bool] = {}
        for model in self.required_models():
            if self.is_downloaded(model):
                log.info("Modell %s bereits vorhanden", model.name)
                results[model.name] = True
                continue
            log.info("Starte Download für %s", model.name)
            results[model.name] = self.download(model, progress_callback=progress_callback)
        return results

    def write_status(self, path: Path | str | None = None) -> None:
        """Persist the current download status for diagnostics."""
        target = Path(path or self._models_dir / "models.json")
        data = {
            m.name: {
                "downloaded": self.is_downloaded(m),
                "path": str(self._model_path(m)),
            }
            for m in self.required_models()
        }
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
