"""The plugin registry.

A flat map from handle to entry, plus a reverse index from capability to the
handle that provides it. The reverse index is the part that matters: it is what
lets the kernel resolve "who produces bars?" without any knowledge of the
plugins that exist.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import Builder, PluginKind, PluginManifest, iter_capabilities
from .errors import DuplicatePlugin, NoProvider, UnknownPlugin


@dataclass(frozen=True, slots=True)
class PluginEntry:
    """A registered plugin: its manifest, its factory, and where it came from."""

    manifest: PluginManifest
    build: Builder
    module: str
    origin: str = "builtin"

    @property
    def handle(self) -> str:
        return self.manifest.handle

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def kind(self) -> PluginKind:
        return self.manifest.kind

    @property
    def is_third_party(self) -> bool:
        return self.origin != "builtin"

    def describe(self) -> dict:
        return {
            "handle": self.handle,
            "kind": self.kind.value,
            "name": self.name,
            "version": self.manifest.version,
            "description": self.manifest.description,
            "provides": list(self.manifest.provides),
            "requires": list(self.manifest.requires),
            "tags": list(self.manifest.tags),
            "origin": self.origin,
            "module": self.module,
        }


class Registry:
    """Handle -> entry, and capability -> handle."""

    def __init__(self) -> None:
        self._entries: dict[str, PluginEntry] = {}
        self._capabilities: dict[str, str] = {}

    # ------------------------------------------------------------- population

    def register(self, entry: PluginEntry) -> PluginEntry:
        handle = entry.handle
        if handle in self._entries:
            existing = self._entries[handle]
            raise DuplicatePlugin(
                f"{handle} is already registered from {existing.module}; "
                f"{entry.module} cannot claim the same handle"
            )
        for capability in entry.manifest.provides:
            owner = self._capabilities.get(capability)
            if owner is not None and owner != handle:
                raise DuplicatePlugin(
                    f"capability {capability!r} is provided by both {owner} and {handle}; "
                    f"a capability must resolve to exactly one plugin"
                )
            self._capabilities[capability] = handle
        self._entries[handle] = entry
        return entry

    def extend(self, entries: Iterator[PluginEntry]) -> list[PluginEntry]:
        return [self.register(entry) for entry in entries]

    # ----------------------------------------------------------------- access

    def get(self, handle: str) -> PluginEntry:
        try:
            return self._entries[handle]
        except KeyError:
            raise UnknownPlugin(self.suggest(handle)) from None

    def find(
        self,
        kind: PluginKind | str | None = None,
        tag: str | None = None,
    ) -> list[PluginEntry]:
        kind_value = kind.value if isinstance(kind, PluginKind) else kind
        found = [
            entry
            for entry in self._entries.values()
            if (kind_value is None or entry.kind.value == kind_value)
            and (tag is None or tag in entry.manifest.tags)
        ]
        return sorted(found, key=lambda entry: entry.handle)

    def provider(self, capability: str) -> str:
        """The handle providing ``capability``.

        Uniqueness is enforced at registration, so this is a lookup rather than a
        search: a capability that resolved to different plugins on different runs
        would make a composition impossible to reason about.
        """
        try:
            return self._capabilities[capability]
        except KeyError:
            raise NoProvider(
                f"no plugin provides {capability!r}. "
                f"Known capabilities: {', '.join(sorted(self._capabilities)) or 'none'}"
            ) from None

    def capabilities(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._capabilities))

    def consumers_of(self, capability: str) -> list[str]:
        """Handles that declare a requirement on ``capability``."""
        return sorted(
            entry.handle
            for entry in self._entries.values()
            if capability in entry.manifest.requires
        )

    # ------------------------------------------------------------- diagnostics

    def suggest(self, handle: str) -> str:
        kind, _, name = handle.partition(":")
        if kind and name:
            near = [
                entry.handle
                for entry in self._entries.values()
                if entry.kind.value == kind and name in entry.name
            ]
            if near:
                return f"unknown plugin {handle!r}; did you mean {', '.join(near)}?"
        if handle in {entry.name for entry in self._entries.values()}:
            matches = [entry.handle for entry in self._entries.values() if entry.name == handle]
            return f"{handle!r} needs a kind prefix: {', '.join(matches)}"
        return (
            f"unknown plugin {handle!r}. "
            f"Registered: {', '.join(sorted(self._entries)) or 'none'}"
        )

    def describe(self) -> list[dict]:
        return [entry.describe() for entry in self.find()]

    # -------------------------------------------------------------- protocols

    def __iter__(self) -> Iterator[PluginEntry]:
        return iter(self.find())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, handle: object) -> bool:
        return str(handle) in self._entries

    def __repr__(self) -> str:
        return f"<Registry {len(self._entries)} plugins, {len(self._capabilities)} capabilities>"


__all__ = ["PluginEntry", "Registry", "iter_capabilities"]
