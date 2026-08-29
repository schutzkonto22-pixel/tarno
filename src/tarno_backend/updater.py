"""Auto-updater for TARNO desktop builds.

Checks a GitHub release endpoint for a newer version and, on Windows, downloads
and stages the installer for the next restart. The actual install step is left
to a small wrapper so the running process can exit cleanly.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    version: str
    url: str
    changelog: str
    checksum_url: str | None = None


class AutoUpdater:
    """Checks GitHub releases for updates and stages installers."""

    def __init__(
        self,
        current_version: str,
        repo: str,
        channel: str = "stable",
    ) -> None:
        self._current_version = current_version
        self._repo = repo
        self._channel = channel

    def check(self) -> UpdateInfo | None:
        """Return UpdateInfo if a newer release is available, else None."""
        try:
            release = self._fetch_latest_release()
            if release is None:
                return None
            version = release.get("tag_name", "").lstrip("v")
            if not version or not self._is_newer(version):
                return None
            asset_url, checksum_url = self._find_asset(release)
            if not asset_url:
                return None
            return UpdateInfo(
                version=version,
                url=asset_url,
                changelog=release.get("body", ""),
                checksum_url=checksum_url,
            )
        except Exception as exc:
            log.warning("Update-Prüfung fehlgeschlagen: %s", exc)
            return None

    def _fetch_latest_release(self) -> dict[str, Any] | None:
        url = f"https://api.github.com/repos/{self._repo}/releases/latest"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def _find_asset(self, release: dict[str, Any]) -> tuple[str | None, str | None]:
        system = platform.system().lower()
        asset_pattern = re.compile(
            r"tarno.*windows.*\.exe" if system == "windows" else r"tarno.*\.zip",
            re.IGNORECASE,
        )
        checksum_pattern = re.compile(
            r"(checksums|sha256sums?|checksums\.txt|sha256sum\.txt)",
            re.IGNORECASE,
        )
        asset_url: str | None = None
        checksum_url: str | None = None
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if asset_pattern.search(name):
                asset_url = asset.get("browser_download_url")
            elif checksum_pattern.search(name):
                checksum_url = asset.get("browser_download_url")
        return asset_url, checksum_url

    def _is_newer(self, version: str) -> bool:
        def normalize(v: str) -> list[int]:
            return [int(x) for x in v.split(".")[:3]]
        try:
            return normalize(version) > normalize(self._current_version)
        except ValueError:
            return False

    def stage_update(self, update: UpdateInfo, staging_dir: Path | str = "~/.tarno/updates") -> Path | None:
        """Download the installer/asset and verify its SHA-256 checksum."""
        dest = Path(staging_dir).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        filename = Path(update.url.split("/")[-1]) or "tarno_update.exe"
        local_path = dest / filename
        try:
            with requests.get(update.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            log.info("Update gestaged: %s", local_path)
        except Exception as exc:
            log.error("Download fehlgeschlagen: %s", exc)
            local_path.unlink(missing_ok=True)
            return None

        if not self._verify_checksum(local_path, update):
            log.error("Update-Checksum-Prüfung fehlgeschlagen; gestagedes Update wird verworfen")
            local_path.unlink(missing_ok=True)
            return None

        return local_path

    def _verify_checksum(self, path: Path, update: UpdateInfo) -> bool:
        """Verify the downloaded file against the release checksum manifest."""
        if not update.checksum_url:
            log.error("Keine Checksum-Datei im Release gefunden; Update wird blockiert.")
            return False
        try:
            response = requests.get(update.checksum_url, timeout=30)
            response.raise_for_status()
            expected = self._extract_checksum(path.name, response.text)
            if expected is None:
                log.error("Kein Checksum-Eintrag für %s gefunden", path.name)
                return False
            actual = self._sha256_of(path)
            if actual.lower() != expected.lower():
                log.error("Checksum mismatch: erwartet %s, erhalten %s", expected, actual)
                return False
            log.info("Update-Checksum OK")
            return True
        except Exception as exc:
            log.error("Checksum-Prüfung fehlgeschlagen: %s", exc)
            return False

    @staticmethod
    def _extract_checksum(filename: str, checksum_text: str) -> str | None:
        for line in checksum_text.splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[1].strip() == filename or Path(parts[1]).name == filename:
                return parts[0].strip()
        return None

    @staticmethod
    def _sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def write_restart_script(installer_path: Path, script_path: Path | str = "~/.tarno/run_update.bat") -> Path:
        """Write a small Windows batch script that runs the installer after exit."""
        path = Path(script_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'@echo off\ntimeout /t 2 /nobreak > nul\nstart "" "{installer_path}" /SILENT\n',
            encoding="utf-8",
        )
        return path
