"""Plugin discovery and lifecycle manager."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import yaml

from tarno_backend.ai.tool_registry import ToolRegistry
from tarno_backend.plugins.plugin import TarnoPlugin

log = logging.getLogger(__name__)


class PluginManager:
    """Discovers, loads and unloads TARNO plugins.

    Plugins are expected to live in directories with a `plugin.yaml` manifest
    and an entrypoint file that exposes a `TarnoPlugin` implementation named
    `plugin`, a `Plugin` class, or a callable `create_plugin()`.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._context = context or {}
        self._plugins: dict[str, TarnoPlugin] = {}
        self._loaded_tools: dict[str, list[str]] = {}
        self._missing_dependencies: dict[str, list[str]] = {}

    def load_from_directory(self, directory: Path | str) -> int:
        """Discover and load all plugins in *directory*. Returns count loaded."""
        dir_path = Path(directory).expanduser()
        if not dir_path.exists():
            log.debug("Plugin-Verzeichnis nicht vorhanden: %s", dir_path)
            return 0

        loaded = 0
        for plugin_dir in sorted(p for p in dir_path.iterdir() if p.is_dir()):
            try:
                if self._load_plugin_dir(plugin_dir):
                    loaded += 1
            except Exception:
                log.exception("Plugin konnte nicht geladen werden: %s", plugin_dir)
        return loaded

    def _load_plugin_dir(self, plugin_dir: Path) -> bool:
        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            manifest_path = plugin_dir / "plugin.yml"
        if not manifest_path.exists():
            log.debug("Kein Manifest in %s", plugin_dir)
            return False

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not self._validate_manifest(plugin_dir.name, manifest):
            return False

        name = manifest["name"]
        entrypoint = manifest.get("entrypoint", "plugin.py")
        entry_path = plugin_dir / entrypoint
        if not entry_path.exists():
            log.warning("Plugin %s hat kein Entrypoint %s", name, entry_path)
            return False

        missing = self._check_dependencies(name, manifest.get("dependencies", []))
        if missing:
            self._missing_dependencies[name] = missing
            log.warning(
                "Plugin %s hat fehlende Abhängigkeiten: %s", name, ", ".join(missing)
            )
            # Continue loading but remember missing deps for diagnostics.

        plugin = self._instantiate(entry_path, manifest)
        if plugin is None:
            return False

        context = {**self._context, "manifest": manifest}
        plugin.load(context=context)
        if not plugin.is_available():
            log.info("Plugin %s ist in dieser Umgebung nicht verfügbar", name)
            plugin.unload()
            return False

        tool_names = []
        for tool in plugin.get_tools():
            self._registry.register(tool)
            tool_names.append(tool.name)

        self._plugins[name] = plugin
        self._loaded_tools[name] = tool_names
        log.info("Plugin geladen: %s (Tools: %s)", name, ", ".join(tool_names))
        return True

    @staticmethod
    def _validate_manifest(plugin_dir_name: str, manifest: dict[str, Any]) -> bool:
        name = manifest.get("name")
        if not name or not isinstance(name, str):
            log.warning("Plugin %s: 'name' fehlt oder ungültig", plugin_dir_name)
            return False
        version = manifest.get("version")
        if not version or not isinstance(version, str):
            log.warning("Plugin %s: 'version' fehlt oder ungültig", name)
            return False
        return True

    @staticmethod
    def _check_dependencies(name: str, dependencies: list[str]) -> list[str]:
        missing: list[str] = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing

    @staticmethod
    def _instantiate(entry_path: Path, manifest: dict[str, Any]) -> TarnoPlugin | None:
        spec = importlib.util.spec_from_file_location(
            f"tarno_dynamic_plugin_{entry_path.parent.name}",
            entry_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Prefer a create_plugin factory, then a top-level Plugin class,
        # then a module attribute named `plugin`.
        if hasattr(module, "create_plugin"):
            return module.create_plugin(manifest)
        if hasattr(module, "Plugin"):
            return module.Plugin()
        if hasattr(module, "plugin") and isinstance(module.plugin, TarnoPlugin):
            return module.plugin

        log.warning("Kein Plugin-Export in %s gefunden", entry_path)
        return None

    def unload(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return
        for tool_name in self._loaded_tools.pop(name, []):
            self._registry.unregister(tool_name)
        plugin.unload()
        log.info("Plugin entladen: %s", name)

    def unload_all(self) -> None:
        for name in list(self._plugins.keys()):
            self.unload(name)

    def get_plugin(self, name: str) -> TarnoPlugin | None:
        return self._plugins.get(name)

    def missing_dependencies(self) -> dict[str, list[str]]:
        return self._missing_dependencies.copy()

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())
