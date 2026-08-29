"""Base class and helpers for production-ready TARNO plugins.

Plugins should subclass BasePlugin, implement `get_tools`, and optionally
override lifecycle hooks. The base class provides safe access to logging,
configuration and audit helpers without coupling the plugin to the TARNO core.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from tarno_backend.ai.tool_registry import ToolDefinition
from tarno_backend.core.action_result import ActionResult


class BasePlugin(ABC):
    """Abstract base class for TARNO plugins.

    A plugin is a self-contained module that provides tools and lifecycle hooks.
    It receives a context dictionary during load, which may contain:

        - config: the relevant subset of TarnoConfig
        - logger: a logger instance
        - memory_store: optional MemoryStore reference
        - audit: optional audit callable
    """

    name: str = ""
    version: str = "0.0.0"
    dependencies: list[str] = []

    def __init__(self) -> None:
        self._context: dict[str, Any] | None = None
        self._logger: logging.Logger = logging.getLogger(self.name or self.__class__.__name__)

    def load(self, context: dict[str, Any] | None = None) -> None:
        """Called by the plugin manager after the plugin is instantiated."""
        self._context = context or {}
        custom_logger = self._context.get("logger")
        if isinstance(custom_logger, logging.Logger):
            self._logger = custom_logger
        self.on_load()

    def unload(self) -> None:
        """Called by the plugin manager before the plugin is removed."""
        self.on_unload()

    def is_available(self) -> bool:
        """Return True if the plugin can function in the current environment."""
        return True

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """Return the ToolDefinitions provided by this plugin."""
        ...

    def on_load(self) -> None:
        """Override to perform setup after the plugin context is available."""

    def on_unload(self) -> None:
        """Override to perform cleanup."""

    def audit(self, action: str, details: dict[str, Any] | None = None) -> None:
        """Emit an audit record if an audit function is configured."""
        if self._context is None:
            return
        audit_fn = self._context.get("audit")
        if callable(audit_fn):
            audit_fn(plugin=self.name, action=action, details=details or {})

    def require_permission(self, permission: str, target: str) -> None:
        """Raise PermissionDeniedError if the plugin lacks permission.

        Subclasses may override this to integrate with a permission service.
        The default implementation allows everything.
        """

    def success(self, message: str, details: dict[str, Any] | None = None) -> ActionResult:
        return ActionResult(success=True, message=message, details=details)

    def failure(self, message: str, error_code: str = "PluginError") -> ActionResult:
        return ActionResult(success=False, message=message, error_code=error_code)
