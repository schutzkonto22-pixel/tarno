"""Permission framework for microphone and external voice usage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable

import yaml

from tarno_backend.core.exceptions import PermissionDeniedError
from tarno_backend.core.command_engine import RiskLevel

log = logging.getLogger(__name__)


class VoicePermissionType(Enum):
    VOICE_INPUT_BASIC = "voice_input_basic"
    VOICE_RECORDING = "voice_recording"
    VOICE_EXTERNAL_OUTPUT = "voice_external_output"
    VOICE_GAME_INTEGRATION = "voice_game_integration"
    VOICE_COMMUNICATION = "voice_communication"


@dataclass(frozen=True)
class VoicePermissionGrant:
    permission: VoicePermissionType
    target: str
    granted_at: datetime
    expires_at: datetime | None
    persistent: bool
    approved_by_user: bool


VoiceConfirmationDialog = Callable[[VoicePermissionType, str, RiskLevel, timedelta | None], tuple[bool, bool]]


class VoicePermissionManager:
    """Manages microphone and external voice permissions.

    - VOICE_INPUT_BASIC is implicitly granted while TARNO is listening.
    - VOICE_RECORDING requires a medium-risk confirmation.
    - VOICE_EXTERNAL_OUTPUT, VOICE_GAME_INTEGRATION and VOICE_COMMUNICATION
      require a high-risk confirmation.
    """

    _RISK_MAP: dict[VoicePermissionType, RiskLevel] = {
        VoicePermissionType.VOICE_INPUT_BASIC: RiskLevel.LOW,
        VoicePermissionType.VOICE_RECORDING: RiskLevel.MEDIUM,
        VoicePermissionType.VOICE_EXTERNAL_OUTPUT: RiskLevel.HIGH,
        VoicePermissionType.VOICE_GAME_INTEGRATION: RiskLevel.HIGH,
        VoicePermissionType.VOICE_COMMUNICATION: RiskLevel.HIGH,
    }

    _DEFAULT_DURATIONS: dict[VoicePermissionType, int | None] = {
        VoicePermissionType.VOICE_INPUT_BASIC: None,
        VoicePermissionType.VOICE_RECORDING: 300,
        VoicePermissionType.VOICE_EXTERNAL_OUTPUT: 600,
        VoicePermissionType.VOICE_GAME_INTEGRATION: 1800,
        VoicePermissionType.VOICE_COMMUNICATION: 600,
    }

    def __init__(
        self,
        policy_file: Path | str | None = None,
        default_durations: dict[str, int | None] | None = None,
        dialog_factory: VoiceConfirmationDialog | None = None,
    ) -> None:
        self._policy_file = Path(policy_file).expanduser() if policy_file else None
        self._dialog_factory = dialog_factory
        self._session_grants: dict[tuple[VoicePermissionType, str], VoicePermissionGrant] = {}
        self._persistent_grants: set[tuple[VoicePermissionType, str]] = set()

        durations: dict[VoicePermissionType, int | None] = {}
        if default_durations:
            for key, value in default_durations.items():
                try:
                    durations[VoicePermissionType(key)] = value
                except ValueError:
                    log.warning("Unbekannte Default-Dauer: %s", key)
        self._default_durations = {**self._DEFAULT_DURATIONS, **durations}
        self._load_persistent()

    @staticmethod
    def risk_level(permission: VoicePermissionType) -> RiskLevel:
        return VoicePermissionManager._RISK_MAP[permission]

    def is_granted(self, permission: VoicePermissionType, target: str) -> bool:
        key = (permission, target)
        if key in self._persistent_grants:
            return True

        grant = self._session_grants.get(key)
        if grant is None:
            return False
        if grant.expires_at is not None and datetime.now() > grant.expires_at:
            self._session_grants.pop(key)
            return False
        return True

    def request(
        self,
        permission: VoicePermissionType,
        target: str,
        duration: timedelta | None = None,
        user_query: str = "",
    ) -> bool:
        if self.is_granted(permission, target):
            return True

        if permission == VoicePermissionType.VOICE_INPUT_BASIC:
            self._grant(permission, target, None, persistent=False)
            return True

        if self._dialog_factory is None:
            raise PermissionDeniedError(
                f"Kein Bestätigungsdialog für {permission.value} konfiguriert"
            )

        risk = self.risk_level(permission)
        if duration is None:
            seconds = self._default_durations.get(permission)
            duration = timedelta(seconds=seconds) if seconds else None

        approved, persistent = self._dialog_factory(permission, target, risk, duration)
        if not approved:
            raise PermissionDeniedError(f"Nutzer hat {permission.value} abgelehnt")

        # High-risk permissions are never persisted automatically.
        if risk == RiskLevel.HIGH:
            persistent = False

        self._grant(permission, target, duration, persistent=persistent)
        return True

    def _grant(
        self,
        permission: VoicePermissionType,
        target: str,
        duration: timedelta | None,
        persistent: bool,
    ) -> None:
        key = (permission, target)
        expires = datetime.now() + duration if duration else None
        grant = VoicePermissionGrant(
            permission=permission,
            target=target,
            granted_at=datetime.now(),
            expires_at=expires,
            persistent=persistent,
            approved_by_user=True,
        )
        self._session_grants[key] = grant
        if persistent:
            self._persistent_grants.add(key)
            self._save_persistent()

    def revoke(self, permission: VoicePermissionType, target: str) -> None:
        key = (permission, target)
        self._session_grants.pop(key, None)
        if key in self._persistent_grants:
            self._persistent_grants.discard(key)
            self._save_persistent()

    def revoke_all(self) -> None:
        self._session_grants.clear()
        self._persistent_grants.clear()
        self._save_persistent()

    def active_grants(self) -> list[VoicePermissionGrant]:
        """Return all currently valid session grants."""
        now = datetime.now()
        valid = []
        for key, grant in list(self._session_grants.items()):
            if grant.expires_at is None or grant.expires_at > now:
                valid.append(grant)
            else:
                self._session_grants.pop(key, None)
        return valid

    def _load_persistent(self) -> None:
        if self._policy_file is None or not self._policy_file.exists():
            return
        try:
            data = yaml.safe_load(self._policy_file.read_text(encoding="utf-8")) or {}
            for item in data.get("persistent_grants", []):
                try:
                    perm = VoicePermissionType(item["permission"])
                    target = item["target"]
                    self._persistent_grants.add((perm, target))
                except (KeyError, ValueError):
                    log.warning("Ungültiger persistent grant: %s", item)
        except Exception:
            log.exception("Fehler beim Laden der Voice-Policy-Datei")

    def _save_persistent(self) -> None:
        if self._policy_file is None:
            return
        self._policy_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "persistent_grants": [
                {"permission": perm.value, "target": target}
                for perm, target in sorted(self._persistent_grants, key=lambda x: (x[0].value, x[1]))
            ]
        }
        self._policy_file.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
