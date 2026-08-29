"""Simple iCalendar reader for proactive briefings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalendarEvent:
    summary: str
    start: datetime | date
    end: datetime | date | None = None


class CalendarService:
    """Reads a local iCalendar (.ics) file and returns events for a range."""

    def __init__(self, calendar_file: str | None = None) -> None:
        self.calendar_file = calendar_file

    def get_events(self, days_ahead: int = 1) -> str:
        """Return a human-readable summary of events for the next N days."""
        if not self.calendar_file:
            return "Kein Kalender konfiguriert."

        path = Path(self.calendar_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return f"Kalenderdatei nicht gefunden: {path}"

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            log.exception("Kalenderdatei konnte nicht gelesen werden")
            return f"Kalender konnte nicht gelesen werden: {exc}"

        events = self._parse_events(text, days_ahead)
        if not events:
            return "Keine Termine im konfigurierten Zeitraum."

        lines = ["Termine:"]
        for ev in events:
            if isinstance(ev.start, datetime):
                start_str = ev.start.strftime("%d.%m.%Y %H:%M")
            else:
                start_str = ev.start.strftime("%d.%m.%Y")
            lines.append(f"- {start_str}: {ev.summary}")
        return "\n".join(lines)

    def get_upcoming_events(self, days_ahead: int = 1) -> list[CalendarEvent]:
        """Return raw CalendarEvent objects for the next N days.

        Used by ProactiveEngine's TimeCalendarObserver (Block 3, Phase 22),
        which needs structured start times to compute "minutes until event"
        rather than the human-readable summary get_events() returns. Returns
        an empty list (never raises) if no calendar is configured, the file
        is missing, or it can't be read - the observer treats that the same
        as "nothing to report" rather than an error.
        """
        if not self.calendar_file:
            return []

        path = Path(self.calendar_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            log.exception("Kalenderdatei konnte nicht gelesen werden")
            return []

        return self._parse_events(text, days_ahead)

    def _parse_events(self, text: str, days_ahead: int) -> list[CalendarEvent]:
        """Parse VEVENT blocks and filter events for the requested range."""
        now = datetime.now()
        start_window = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_window = start_window + timedelta(days=max(days_ahead, 1))
        events: list[CalendarEvent] = []

        # Fold continuation lines (lines that start with a space or tab)
        folded = re.sub(r"\r?\n[ \t]", "", text)
        # Find all VEVENT blocks
        for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", folded, re.DOTALL | re.IGNORECASE):
            start = self._parse_dt(_extract_field(block, "DTSTART"))
            summary = _extract_field(block, "SUMMARY") or "(kein Titel)"
            if start is None:
                continue

            if isinstance(start, datetime):
                in_range = start < end_window and start >= start_window
            else:
                in_range = start < end_window.date() and start >= start_window.date()

            if in_range:
                end = self._parse_dt(_extract_field(block, "DTEND"))
                events.append(CalendarEvent(summary=summary, start=start, end=end))

        events.sort(key=lambda e: e.start)
        return events

    def _parse_dt(self, value: str | None) -> datetime | date | None:
        """Parse an iCalendar DTSTART/DTEND value."""
        if not value:
            return None

        # Remove parameters like TZID=... or VALUE=DATE:
        if ":" in value:
            value = value.rsplit(":", 1)[-1]

        # All-day date
        if len(value) == 8 and value.isdigit():
            return date(int(value[:4]), int(value[4:6]), int(value[6:8]))

        # Floating date-time (local) or UTC
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                dt = datetime.strptime(value.replace("Z", ""), fmt)
                if value.endswith("Z"):
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue

        return None


def _extract_field(block: str, field: str) -> str | None:
    """Extract a single field value from an iCalendar block."""
    match = re.search(rf"^{re.escape(field)}[^\n]*:(.*?)$", block, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def get_calendar_events(calendar_file: str, days_ahead: int = 1) -> str:
    """Convenience helper for the tool registry."""
    return CalendarService(calendar_file).get_events(days_ahead)
