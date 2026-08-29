"""Audit-log retention and integrity checks for TARNO."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


class AuditManager:
    """Manages retention and integrity of audit log files."""

    def __init__(self, log_dir: Path | str = "~/.tarno/logs") -> None:
        self._log_dir = Path(log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._log_dir / ".audit_state"

    def rotate_old_logs(self, retention_days: int = 365) -> int:
        """Delete audit logs older than *retention_days*. Returns count."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        removed = 0
        for path in self._log_dir.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
                log.info("Audit-Log rotiert: %s", path.name)
        return removed

    def current_hash_chain(self) -> list[tuple[Path, str]]:
        """Return SHA-256 hashes of all audit files for integrity checks."""
        result: list[tuple[Path, str]] = []
        for path in sorted(self._log_dir.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result.append((path, digest))
        return result

    def save_state(self) -> None:
        """Store the current audit file hashes so tampering can be detected later."""
        state = {
            "saved_at": datetime.now().isoformat(),
            "files": [
                {"name": p.name, "hash": h}
                for p, h in self.current_hash_chain()
            ],
        }
        import json
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify current audit files against the saved state.

        Returns (ok, list_of_violations).
        """
        import json
        if not self._state_file.exists():
            return True, []

        state = json.loads(self._state_file.read_text(encoding="utf-8"))
        expected = {item["name"]: item["hash"] for item in state.get("files", [])}
        current = {p.name: h for p, h in self.current_hash_chain()}

        violations: list[str] = []
        for name, expected_hash in expected.items():
            if name not in current:
                violations.append(f"{name} wurde gelöscht")
            elif current[name] != expected_hash:
                violations.append(f"{name} wurde verändert")
        for name in current:
            if name not in expected:
                violations.append(f"{name} wurde hinzugefügt")

        return not violations, violations

    def export_for_retention(self, target_dir: Path | str, older_than_days: int = 30) -> list[Path]:
        """Copy logs older than *older_than_days* into *target_dir* for archiving."""
        target = Path(target_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now() - timedelta(days=older_than_days)
        exported: list[Path] = []
        for path in self._log_dir.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                dest = target / path.name
                import shutil
                shutil.copy2(path, dest)
                exported.append(dest)
        return exported
