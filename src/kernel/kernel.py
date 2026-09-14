"""The kernel: registers plugins, builds them, and wires them by capability.

Building is lazy and memoised. Asking for a handle twice returns the same
instance — a feed or a model must not be constructed twice — while asking with
explicit params builds a fresh object, which is what a test or a second
simulation wants. The rule is one line long, and it removes the usual
"is this thing a singleton?" ambiguity from plugin code entirely.

``validate`` is the other half of the contract: before a composition runs, every
declared requirement is checked against a registered provider, so a missing
plugin is reported as a wiring error at startup rather than as a ``None`` that
blows up mid-session.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from core.settings import Settings

from .bus import EventBus
from .context import PluginContext
from .contracts import PluginKind, PluginManifest
from .errors import MissingDependency, UnknownPlugin
from .loader import LoadReport, discover, discover_entry_points
from .registry import PluginEntry, Registry


class Kernel:
    """Composition root for the plugin graph."""

    def __init__(
        self,
        settings: Settings,
        registry: Registry | None = None,
        bus: EventBus | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or Registry()
        self.bus = bus or EventBus()
        self.log = logger or logging.getLogger("kernel")
        self.load_report = LoadReport()
        self._instances: dict[str, Any] = {}
        self._params: dict[str, dict[str, Any]] = {
            handle: dict(params) for handle, params in (settings.plugin_config or {}).items()
        }

    # --------------------------------------------------------------- assembly

    @classmethod
    def bootstrap(
        cls,
        settings: Settings,
        *,
        with_entry_points: bool = True,
        logger: logging.Logger | None = None,
    ) -> Kernel:
        """Discover and register every available plugin."""
        kernel = cls(settings, logger=logger)
        kernel.load_report = discover()
        if with_entry_points:
            kernel.load_report.extend(discover_entry_points())
        for entry in kernel.load_report.entries:
            kernel.registry.register(entry)
        for module, error in kernel.load_report.errors:
            kernel.log.warning("plugin %s failed to load: %s", module, error)
        return kernel

    def register(self, entry: PluginEntry) -> PluginEntry:
        return self.registry.register(entry)

    # ---------------------------------------------------------------- building

    def params_for(self, handle: str) -> dict[str, Any]:
        """Configured params for a handle, normally from its nested YAML file."""
        return dict(self._params.get(handle, {}))

    def configure(self, params: Mapping[str, Mapping[str, Any]]) -> None:
        for handle, values in params.items():
            self._params.setdefault(handle, {}).update(dict(values))

    def build(self, handle: str, **params: Any) -> Any:
        """Build a plugin. Cached when unparameterised, fresh when parameterised."""
        entry = self.registry.get(handle)
        if params:
            return self._invoke(entry, {**self.params_for(handle), **params})
        if handle not in self._instances:
            self._instances[handle] = self._invoke(entry, self.params_for(handle))
        return self._instances[handle]

    def new(self, handle: str, **params: Any) -> Any:
        """Build a fresh instance, ignoring the cache.

        ``build`` memoises because a feed or a model must not be constructed
        twice. This is the opposite request, and it is needed the moment there is
        more than one of something: three instruments on one board need three
        aggregators and three projections, each with its own state, sharing one
        kernel. Configured params are still applied underneath the explicit ones.
        """
        entry = self.registry.get(handle)
        return self._invoke(entry, {**self.params_for(handle), **params})

    def capability(self, name: str) -> Any:
        """Build whichever plugin provides ``name``."""
        return self.build(self.registry.provider(name))

    def provider(self, name: str) -> str:
        return self.registry.provider(name)

    def _invoke(self, entry: PluginEntry, params: Mapping[str, Any]) -> Any:
        merged = {**dict(entry.manifest.params), **params}
        context = PluginContext(
            _kernel=self,
            bus=self.bus,
            registry=self.registry,
            params=dict(merged),
            logger=self.log.getChild(entry.handle.replace(":", ".")),
        )
        try:
            return entry.build(context, **merged)
        except Exception as exc:  # noqa: BLE001 — re-raise with the handle attached
            raise type(exc)(f"{entry.handle} failed to build: {exc}") from exc

    def built(self) -> dict[str, Any]:
        """Handles that have already been instantiated."""
        return dict(self._instances)

    def teardown(self) -> None:
        """Stop everything that was built and drop the event bus."""
        for handle, instance in list(self._instances.items()):
            stop = getattr(instance, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("plugin %s failed to stop: %s", handle, exc)
        self._instances.clear()
        self.bus.clear()

    # ------------------------------------------------------------ diagnostics

    def validate(self, handles: Iterable[str] | None = None) -> list[str]:
        """Check that every requirement of the selected plugins is satisfied.

        Returns the problems rather than raising them so a caller can report all
        of them at once — a missing capability is usually one line in a config
        file, and seeing every consequence of it in a single pass is what makes
        it quick to fix.
        """
        selected = list(handles) if handles is not None else [entry.handle for entry in self.registry]
        problems: list[str] = []
        capabilities = self.registry.capabilities()
        for handle in selected:
            try:
                entry = self.registry.get(handle)
            except UnknownPlugin as exc:
                problems.append(str(exc))
                continue
            for requirement in entry.manifest.requires:
                if requirement not in capabilities:
                    problems.append(
                        f"{handle} requires {requirement!r}, which no registered plugin provides"
                    )
        return problems

    def require(self, handles: Iterable[str] | None = None) -> None:
        problems = self.validate(handles)
        if problems:
            raise MissingDependency("; ".join(problems))

    def entries(self, kind: PluginKind | str | None = None) -> list[PluginEntry]:
        return self.registry.find(kind)

    def describe(self) -> list[dict]:
        return self.registry.describe()

    def manifest(self, handle: str) -> PluginManifest:
        return self.registry.get(handle).manifest

    def __repr__(self) -> str:
        return f"<Kernel {len(self.registry)} plugins, {self.bus.listeners()} listeners>"


def bootstrap(settings: Settings, **kwargs: Any) -> Kernel:
    """Convenience wrapper for ``Kernel.bootstrap``."""
    return Kernel.bootstrap(settings, **kwargs)


__all__ = ["Kernel", "bootstrap"]
