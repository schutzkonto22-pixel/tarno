"""Privacy controls for TARNO memory and conversation data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tarno_backend.memory.store import MemoryStore

log = logging.getLogger(__name__)


class MemoryPrivacy:
    """Provides GDPR-style deletion, export and retention management."""

    def __init__(
        self,
        store: MemoryStore,
        recordings_dir: Path | str = "~/.tarno/recordings",
        logs_dir: Path | str = "~/.tarno/logs",
    ) -> None:
        self._store = store
        self._recordings_dir = Path(recordings_dir).expanduser()
        self._logs_dir = Path(logs_dir).expanduser()

    def export_memory(self, target_path: Path | str) -> Path:
        """Export all memory to a JSON file before deletion."""
        target = Path(target_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "exported_at": datetime.now().isoformat(),
            "facts": [],
            "episodes": [],
            "preferences": [],
        }
        # MemoryStore does not expose raw dump, so we rely on search.
        for fact in self._store.search_facts("", limit=10000):
            data["facts"].append({
                "key": fact.key,
                "value": fact.value,
                "source": fact.source,
                "updated_at": fact.updated_at.isoformat(),
            })
        for episode in self._store.get_recent_episodes(limit=10000):
            data["episodes"].append({
                "summary": episode.summary,
                "tags": episode.tags,
                "created_at": episode.created_at.isoformat(),
            })
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Memory exportiert nach %s", target)
        return target

    def delete_all(self) -> None:
        """Delete all memory data."""
        self._store.delete_all()
        log.info("Gesamtes Memory gelöscht")

    def delete_facts_by_key(self, key: str) -> None:
        self._store.delete_by_key(key)
        log.info("Fakten/Präferenzen mit key=%s gelöscht", key)

    def delete_recordings(self, older_than_days: int | None = None) -> int:
        """Delete recordings optionally older than a threshold. Returns count."""
        if not self._recordings_dir.exists():
            return 0
        cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None
        removed = 0
        for path in self._recordings_dir.iterdir():
            if not path.is_file():
                continue
            if cutoff is None or datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        log.info("%d Aufnahmen gelöscht", removed)
        return removed

    def delete_old_logs(self, retention_days: int = 365) -> int:
        """Delete audit logs older than retention_days. Returns count."""
        if not self._logs_dir.exists():
            return 0
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0
        for path in self._logs_dir.iterdir():
            if not path.is_file():
                continue
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        log.info("%d alte Logs gelöscht", removed)
        return removed
