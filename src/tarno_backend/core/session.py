"""Session and turn context management for TARNO."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tarno_backend.telemetry.logging import set_correlation_id


@dataclass
class TurnContext:
    """Mutable context for a single user turn."""

    turn_id: str
    started_at: datetime = field(default_factory=datetime.now)
    user_text: str = ""
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Container for one continuous TARNO session."""

    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    turns: list[TurnContext] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def new_turn(self, user_text: str = "") -> TurnContext:
        """Create a new turn in this session and set its correlation ID."""
        turn_id = str(uuid.uuid4())
        set_correlation_id(turn_id)
        turn = TurnContext(turn_id=turn_id, user_text=user_text)
        self.turns.append(turn)
        return turn

    def current_turn(self) -> TurnContext | None:
        return self.turns[-1] if self.turns else None


class SessionManager:
    """Owns the active session and provides factory methods."""

    def __init__(self) -> None:
        self._session: Session | None = None

    def start_session(self, metadata: dict[str, Any] | None = None) -> Session:
        """Start a new session, replacing any previous one."""
        session_id = str(uuid.uuid4())
        set_correlation_id(session_id)
        self._session = Session(
            session_id=session_id,
            metadata=metadata or {},
        )
        return self._session

    def end_session(self) -> None:
        """Close the active session and clear its correlation ID."""
        self._session = None
        set_correlation_id(None)

    @property
    def session(self) -> Session | None:
        return self._session

    def ensure_session(self) -> Session:
        if self._session is None:
            return self.start_session()
        return self._session
