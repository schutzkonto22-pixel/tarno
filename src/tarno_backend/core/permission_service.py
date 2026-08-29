"""Permission service for command and voice actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

from tarno_backend.core.command_engine import PreparedCommand, RiskLevel
from tarno_backend.core.exceptions import PermissionDeniedError

log = logging.getLogger(__name__)


class AutonomyMode(Enum):
    """Live-switchable autonomy level, analogous to Claude Code's permission modes.

    MANUAL: LOW auto (always was, pure info queries), MEDIUM/HIGH always confirm.
    BALANCED (default): LOW+MEDIUM auto (MEDIUM only announced via speech,
        never blocked on a dialog), HIGH still requires the dialog + safety phrase.
    PLAN: nothing executes for real without an explicit confirmation of the plan
        (see CommandTool, which forces dry_run=True in this mode).
    ULTIMATE: LOW+MEDIUM+HIGH all auto, including skipping the HIGH safety phrase.

    None of these modes ever touch the RiskLevel.BLOCKED tier - that check happens
    earlier in CommandEngine.prepare(), before PermissionService is even called, and
    is intentionally mode-independent (see tests/test_risk_pipeline.py).
    """

    MANUAL = "manual"
    BALANCED = "balanced"
    PLAN = "plan"
    ULTIMATE = "ultimate"


class PermissionType(Enum):
    COMMAND_LOW = "command_low"
    COMMAND_MEDIUM = "command_medium"
    COMMAND_HIGH = "command_high"
    VOICE_INPUT_BASIC = "voice_input_basic"
    VOICE_RECORDING = "voice_recording"
    VOICE_EXTERNAL_OUTPUT = "voice_external_output"
    VOICE_GAME_INTEGRATION = "voice_game_integration"
    VOICE_COMMUNICATION = "voice_communication"
    CODING_EDIT = "coding_edit"
    CODING_SHELL = "coding_shell"


@dataclass
class PermissionGrant:
    permission: PermissionType
    target: str
    granted_at: datetime
    expires_at: datetime | None
    persistent: bool


# Risk-specific confirmation factory signature.
ConfirmationDialog = Callable[[PermissionType, str, RiskLevel, timedelta | None], tuple[bool, bool]]


class PermissionService:
    """Manages user approvals for risky actions.

    Low-risk actions are auto-approved. Medium and high risk actions show a
    confirmation dialog. High-risk actions additionally require a typed safety
    phrase. Approved medium-risk actions may be cached for the current session.
    """

    _SAFETY_PHRASE = "Ja, ausführen"

    def __init__(
        self,
        dialog_factory: ConfirmationDialog | None = None,
        mode: AutonomyMode = AutonomyMode.BALANCED,
    ) -> None:
        self._dialog_factory = dialog_factory
        # None = use the built-in console input() prompt (default/legacy behavior).
        self._safety_phrase_confirm: Callable[[], bool] | None = None
        self._session_grants: dict[tuple[PermissionType, str], PermissionGrant] = {}
        self._mode = mode

    @property
    def mode(self) -> AutonomyMode:
        return self._mode

    def set_mode(self, mode: AutonomyMode) -> None:
        log.info("Autonomie-Modus geändert: %s -> %s", self._mode.value, mode.value)
        self._mode = mode

    def set_dialog_factory(
        self,
        dialog_factory: ConfirmationDialog,
        safety_phrase_confirm: Callable[[], bool] | None = None,
    ) -> None:
        """Swap the confirmation dialog implementation after construction.

        Used by the gRPC bridge to replace the Qt/console default with one
        that round-trips the confirmation to the connected WinUI frontend,
        since the gRPC backend process has neither a Qt event loop nor an
        attached console to fall back to.

        *safety_phrase_confirm*, if given, replaces the console `input()`
        prompt normally used for the extra HIGH-risk safety-phrase check
        (e.g. because the WinUI dialog already enforced typing the phrase
        as part of the same round-trip as *dialog_factory*).
        """
        self._dialog_factory = dialog_factory
        if safety_phrase_confirm is not None:
            self._safety_phrase_confirm = safety_phrase_confirm

    def request_approval(
        self,
        prepared: PreparedCommand,
        duration: timedelta | None = None,
    ) -> bool:
        """Return True if the prepared command is approved for execution."""
        permission = self._map_risk_to_permission(prepared.risk_level)
        target = f"{prepared.intent}:{prepared.command}"

        if self._is_granted(permission, target):
            return True

        if self._auto_grants(prepared.risk_level):
            self._grant(permission, target, duration, persistent=False)
            return True

        if self._dialog_factory is None:
            log.error("Kein Bestätigungsdialog konfiguriert für %s", target)
            raise PermissionDeniedError(f"Bestätigungsdialog fehlt für {target}")

        approved, persistent = self._dialog_factory(
            permission,
            prepared.command,
            prepared.risk_level,
            duration,
        )

        if not approved:
            raise PermissionDeniedError(f"Nutzer hat {target} abgelehnt")

        # High-risk always requires an extra safety phrase and is never persistent.
        if prepared.risk_level == RiskLevel.HIGH:
            confirm_phrase = self._safety_phrase_confirm or self._confirm_high_risk_safety_phrase
            if not confirm_phrase():
                raise PermissionDeniedError("Sicherheitsabfrage nicht bestätigt")
            persistent = False

        self._grant(permission, target, duration, persistent=persistent)
        return True

    def request_coding_edit(self, prompt: str, auto_allow: bool = False) -> None:
        """Raise PermissionDeniedError if the user does not approve a coding edit.

        When auto_allow is True, the edit is silently permitted.  In all other
        cases a confirmation dialog is shown (or the console fallback).  This is
        intentionally a separate method from request_approval() because coding
        edits are not Shell-PreparedCommands and need a different target string.
        """
        if auto_allow or self._mode == AutonomyMode.ULTIMATE:
            return

        permission = PermissionType.CODING_EDIT
        target = prompt[:120]

        if self._is_granted(permission, target):
            return

        if self._dialog_factory is None:
            log.error("Kein Bestätigungsdialog konfiguriert für Coding-Edit")
            raise PermissionDeniedError("Bestätigungsdialog fehlt für Coding-Edit")

        approved, persistent = self._dialog_factory(
            permission,
            target,
            RiskLevel.HIGH,
            None,
        )
        if not approved:
            raise PermissionDeniedError("Nutzer hat Coding-Edit abgelehnt")

        self._grant(permission, target, None, persistent=persistent)

    def _auto_grants(self, risk: RiskLevel) -> bool:
        """Whether the current AutonomyMode auto-approves this risk level without a
        confirmation dialog. Never consulted for RiskLevel.BLOCKED - that tier never
        reaches PermissionService at all (see CommandEngine.prepare(), which raises
        CommandBlockedError before this method could ever run). This is the
        architectural guarantee that no AutonomyMode - including ULTIMATE - can
        bypass the hard BLOCKED floor (see tests/test_risk_pipeline.py).
        """
        if self._mode == AutonomyMode.PLAN:
            # CommandTool forces dry_run=True in PLAN mode, so nothing real
            # executes here regardless of risk level - safe to auto-grant.
            return True
        if self._mode == AutonomyMode.ULTIMATE:
            # Bypasses even the HIGH safety phrase (skipped entirely below, since
            # this returns True before the dialog_factory/safety-phrase branch runs).
            return True
        if risk == RiskLevel.LOW:
            return True
        if risk == RiskLevel.MEDIUM:
            return self._mode == AutonomyMode.BALANCED
        return False  # HIGH under MANUAL/BALANCED always needs the dialog + safety phrase.

    def _is_granted(self, permission: PermissionType, target: str) -> bool:
        key = (permission, target)
        grant = self._session_grants.get(key)
        if grant is None:
            return False
        if grant.expires_at is not None and datetime.now() > grant.expires_at:
            self._session_grants.pop(key, None)
            return False
        return True

    def _grant(
        self,
        permission: PermissionType,
        target: str,
        duration: timedelta | None,
        persistent: bool,
    ) -> None:
        expires = datetime.now() + duration if duration else None
        self._session_grants[(permission, target)] = PermissionGrant(
            permission=permission,
            target=target,
            granted_at=datetime.now(),
            expires_at=expires,
            persistent=persistent,
        )

    def revoke_all(self) -> None:
        self._session_grants.clear()

    @staticmethod
    def _map_risk_to_permission(risk: RiskLevel) -> PermissionType:
        mapping = {
            RiskLevel.LOW: PermissionType.COMMAND_LOW,
            RiskLevel.MEDIUM: PermissionType.COMMAND_MEDIUM,
            RiskLevel.HIGH: PermissionType.COMMAND_HIGH,
        }
        return mapping.get(risk, PermissionType.COMMAND_MEDIUM)

    @staticmethod
    def _confirm_high_risk_safety_phrase() -> bool:
        """Console fallback for the high-risk safety phrase."""
        try:
            answer = input(
                f"Sicherheitsabfrage: Geben Sie '{PermissionService._SAFETY_PHRASE}' ein, "
                "um fortzufahren: "
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip() == PermissionService._SAFETY_PHRASE
