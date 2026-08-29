"""TARNO plugin for smart home device control."""

from __future__ import annotations

from typing import Any

from tarno_backend.ai.tool_registry import ToolDefinition
from tarno_backend.core.action_result import ActionResult
from tarno_backend.integrations.smart_home.client import (
    HomeAssistantBackend,
    SmartHomeClient,
)
from tarno_backend.plugins.base import BasePlugin


class SmartHomePlugin(BasePlugin):
    """Provides discovery and control tools for smart home devices."""

    name = "smart_home"
    version = "1.0.0"
    dependencies: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self._client = SmartHomeClient()

    def on_load(self) -> None:
        config = self._context.get("config") if self._context else {}
        if isinstance(config, dict) and config.get("home_assistant"):
            ha = config["home_assistant"]
            backend = HomeAssistantBackend(
                base_url=ha.get("base_url", "http://homeassistant.local:8123"),
                token=ha.get("token", ""),
            )
            self._client.add_backend(backend)
        self.audit("loaded", {"backends": len(self._client._backends)})

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="smart_home_discover",
                description="Listet verfügbare Smart-Home-Geräte auf.",
                input_schema={"type": "object", "properties": {}},
                handler=self._discover,
            ),
            ToolDefinition(
                name="smart_home_set_state",
                description="Schaltet ein Smart-Home-Gerät (z.B. Licht an/aus).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "key": {"type": "string", "default": "state"},
                        "value": {},
                    },
                    "required": ["device_id", "value"],
                },
                handler=self._set_state,
            ),
        ]

    def _discover(self) -> ActionResult:
        try:
            devices = self._client.discover()
            if not devices:
                return self.success("Keine Smart-Home-Geräte gefunden.")
            lines = [f"{d.name} ({d.device_type}) in {d.room}: {d.state}" for d in devices[:20]]
            return self.success("\n".join(lines), details={"count": len(devices)})
        except Exception as exc:
            return self.failure(str(exc))

    def _set_state(self, device_id: str, value: Any, key: str = "state") -> ActionResult:
        ok = self._client.set_state(device_id, key, value)
        return self.success(f"{device_id} gesetzt") if ok else self.failure("Steuerung fehlgeschlagen")


def create_plugin(_manifest: dict[str, Any]) -> SmartHomePlugin:
    return SmartHomePlugin()
