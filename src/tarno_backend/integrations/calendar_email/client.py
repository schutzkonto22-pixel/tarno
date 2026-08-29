"""Calendar and email client wrappers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    summary: str
    start: datetime
    end: datetime
    location: str


class CalendarClient:
    """Reads calendar events from an ICS file."""

    def __init__(self, calendar_file: Path | str | None = None) -> None:
        self._calendar_file = Path(calendar_file).expanduser() if calendar_file else None

    def list_events(self, days: int = 1) -> list[CalendarEvent]:
        from tarno_backend.core.calendar_service import get_calendar_events

        if not self._calendar_file or not self._calendar_file.exists():
            return []
        text = get_calendar_events(str(self._calendar_file), days_ahead=days)
        events: list[CalendarEvent] = []
        # Simple parsing of the formatted text returned by the existing service.
        # A real implementation would parse the ICS structure directly.
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            line = line.lstrip("- ").strip()
            # Format: "dd.mm.yyyy hh:mm: summary" (summary may contain colons)
            match = re.match(
                r"^(\d{2}\.\d{2}\.\d{4}(?:\s+\d{2}:\d{2})?):\s*(.*)$",
                line,
            )
            if not match:
                continue
            time_part, summary = match.group(1), match.group(2)
            try:
                start = datetime.strptime(time_part.strip(), "%d.%m.%Y %H:%M")
            except ValueError:
                try:
                    start = datetime.strptime(time_part.strip(), "%d.%m.%Y")
                except ValueError:
                    continue
            events.append(CalendarEvent(
                summary=summary.strip(),
                start=start,
                end=start + timedelta(hours=1),
                location="",
            ))
        return events


class EmailClient:
    """Skeleton for an IMAP/SMTP email client."""

    def __init__(self, host: str, username: str, password: str, port: int = 993) -> None:
        self._host = host
        self._username = username
        self._password = password
        self._port = port

    def is_available(self) -> bool:
        try:
            import imaplib
            with imaplib.IMAP4_SSL(self._host, self._port) as _:
                return True
        except Exception:
            return False

    def unread_count(self) -> int:
        try:
            import imaplib
            with imaplib.IMAP4_SSL(self._host, self._port) as mail:
                mail.login(self._username, self._password)
                mail.select("inbox")
                status, messages = mail.search(None, "UNSEEN")
                if status != "OK":
                    return 0
                return len(messages[0].split())
        except Exception as exc:
            log.warning("E-Mail-Abfrage fehlgeschlagen: %s", exc)
            return 0
