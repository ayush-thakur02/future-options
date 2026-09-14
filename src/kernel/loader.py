"""Plugin discovery.

Two sources, one shape:

1. **Bundled packs** — every ``plugin.py`` anywhere under ``plugins``,
   found by walking the package. Nested arbitrarily deep, so a plugin can be a
   folder of related modules or a single file; the loader does not care.
2. **Third-party distributions** — anything advertising the
   ``plugins`` entry-point group, so a plugin can live in its own
   installed package.

A plugin that fails to import does not abort discovery. The failure is recorded
in the :class:`LoadReport` and reported alongside the rest of the wiring, because
one broken third-party pack should not stop the platform from trading its own data.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from .contracts import Builder, PluginManifest
from .errors import PluginLoadError
from .registry import PluginEntry

BUILTIN_PACKAGE = "plugins"
ENTRY_POINT_GROUP = "plugins"
PLUGIN_MODULE = "plugin"


@dataclass(slots=True)
class LoadReport:
    """What discovery found, and what it could not load."""

    entries: list[PluginEntry] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, entry: PluginEntry) -> None:
        self.entries.append(entry)

    def fail(self, module: str, error: Exception) -> None:
        self.errors.append((module, str(error)))

    def extend(self, other: LoadReport) -> LoadReport:
        self.entries.extend(other.entries)
        self.errors.extend(other.errors)
        return self

    def summary(self) -> dict:
        return {
            "plugins": len(self.entries),
            "errors": [{"module": module, "error": error} for module, error in self.errors],
        }


def discover(package: str = BUILTIN_PACKAGE, report: LoadReport | None = None) -> LoadReport:
    """Find every ``plugin.py`` under ``package``."""
    report = report or LoadReport()
    try:
        root = importlib.import_module(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - the package always exists
        report.fail(package, exc)
        return report

    for module_name in _plugin_modules(root, package, report):
        try:
            report.add(load_module(module_name))
        except Exception as exc:  # noqa: BLE001 — a bad plugin is data, not a crash
            report.fail(module_name, exc)
    return report


def discover_entry_points(group: str = ENTRY_POINT_GROUP) -> LoadReport:
    """Find plugins advertised by installed distributions."""
    report = LoadReport()
    for module_name in _entry_point_modules(group):
        try:
            report.add(load_module(module_name, origin="entry-point"))
        except Exception as exc:  # noqa: BLE001
            report.fail(module_name, exc)
    return report


def load_module(module_name: str, origin: str = "builtin") -> PluginEntry:
    """Import a plugin module and turn it into a registry entry."""
    module = importlib.import_module(module_name)
    return entry_from_module(module, origin=origin)


def entry_from_module(module: ModuleType, origin: str = "builtin") -> PluginEntry:
    manifest = read_manifest(module)
    builder = read_builder(module)
    return PluginEntry(
        manifest=manifest,
        build=builder,
        module=module.__name__,
        origin=origin,
    )


def read_manifest(module: ModuleType) -> PluginManifest:
    payload = getattr(module, "MANIFEST", None)
    if payload is None:
        raise PluginLoadError(
            f"{module.__name__} does not define MANIFEST. A plugin needs a manifest "
            f"(name, kind, provides) and a build(ctx, **params) function."
        )
    if isinstance(payload, PluginManifest):
        return payload
    if isinstance(payload, Mapping):
        return PluginManifest.from_mapping(payload)
    raise PluginLoadError(
        f"{module.__name__}.MANIFEST must be a PluginManifest or a mapping, "
        f"got {type(payload).__name__}"
    )


def read_builder(module: ModuleType) -> Builder:
    builder: Any = getattr(module, "build", None)
    if builder is None or not callable(builder):
        raise PluginLoadError(f"{module.__name__} does not define a callable build(ctx, **params)")
    return builder


def _plugin_modules(root: ModuleType, package: str, report: LoadReport) -> Iterator[str]:
    """Every importable module named ``plugin`` under ``package``.

    ``walk_packages`` needs each plugin folder to be a real package, which is why
    every level of the tree carries an ``__init__.py``. That is a small price for
    arbitrary nesting, and it keeps a plugin folder importable on its own.

    ``onerror`` is not optional. Without it ``walk_packages`` swallows an
    ImportError from a subpackage and simply does not descend into it, so a pack
    with a broken import disappears from the build with nothing reported — the
    exact opposite of what this module promises. Re-importing inside the callback
    is what turns "something failed" into the actual traceback, which is the part
    a reader needs.
    """
    paths = list(getattr(root, "__path__", []))

    def note(name: str) -> None:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — this is the report path
            report.fail(name, exc)
        else:  # pragma: no cover - the import succeeded on retry, so say so plainly
            report.fail(name, PluginLoadError(f"{name} could not be scanned for plugins"))

    discovered: list[str] = []
    for module in pkgutil.walk_packages(paths, prefix=f"{package}.", onerror=note):
        if module.name.rsplit(".", 1)[-1] == PLUGIN_MODULE:
            discovered.append(module.name)
    yield from sorted(discovered)


def _entry_point_modules(group: str) -> Iterator[str]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python 3.11 always has it
        return

    try:
        selected = entry_points(group=group)
    except TypeError:  # pragma: no cover - importlib API drift
        selected = entry_points().get(group, [])  # type: ignore[attr-defined]
    for point in selected:
        yield point.value if hasattr(point, "value") else str(point)


__all__ = [
    "BUILTIN_PACKAGE",
    "ENTRY_POINT_GROUP",
    "PLUGIN_MODULE",
    "LoadReport",
    "discover",
    "discover_entry_points",
    "entry_from_module",
    "load_module",
]
