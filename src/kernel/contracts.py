"""What a plugin is.

A plugin is a package containing a ``plugin.py`` module that exposes exactly two
names:

    MANIFEST = PluginManifest(name=..., kind=..., provides=(...), requires=(...))
    def build(ctx: PluginContext, **params) -> object: ...

That is the entire contract. There is no base class to inherit, no decorator to
remember, and no import of the kernel required beyond the manifest type. The
cheapest useful plugin is one file.

**Plugins link through capabilities, not imports.** A manifest declares the
capabilities it ``provides`` ("bars", "conviction") and the ones it ``requires``.
Nothing ever imports another plugin's module, so any provider can be swapped for
another that offers the same capability, and the kernel can prove a composition
is complete before anything runs. That is what makes the tree plug-and-play
rather than merely modular.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class PluginKind(StrEnum):
    """The slot a plugin occupies in the pipeline.

    Kinds exist so the CLI can group plugins and so a handle reads as
    ``source:upstox`` rather than a bare name that collides across roles.
    """

    SOURCE = "source"
    AGGREGATOR = "aggregator"
    FEATURES = "features"
    STRATEGY = "strategy"
    FORECAST = "forecast"
    RENDERER = "renderer"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Everything the kernel knows about a plugin before building it."""

    name: str
    kind: PluginKind
    description: str = ""
    version: str = "0.1.0"
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the declarative form a plugin author actually writes.

        ``kind="source"`` and ``provides=["bars"]`` are what a plugin.py looks
        like when it is written to be read. Coercing here means the rest of the
        kernel only ever sees a ``PluginKind`` and tuples, so a str kind cannot
        survive to fail later at ``.value`` — which is exactly what it did before
        this existed.
        """
        object.__setattr__(self, "kind", PluginKind(self.kind))
        object.__setattr__(self, "provides", tuple(self.provides))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def handle(self) -> str:
        return f"{self.kind.value}:{self.name}"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PluginManifest:
        """Accept a dict as well as a manifest, so a plugin stays declarative."""
        return cls(
            name=str(payload["name"]),
            kind=PluginKind(payload["kind"]),
            description=str(payload.get("description", "")),
            version=str(payload.get("version", "0.1.0")),
            provides=tuple(str(item) for item in payload.get("provides", ())),
            requires=tuple(str(item) for item in payload.get("requires", ())),
            tags=tuple(str(item) for item in payload.get("tags", ())),
            params=dict(payload.get("params", {})),
        )


@runtime_checkable
class KernelPort(Protocol):
    """The slice of the kernel a plugin is allowed to use.

    Narrow on purpose: a plugin can build another plugin and resolve a
    capability, and that is all. It cannot reach into the registry to mutate it,
    which keeps the graph a tree built from the leaves up instead of a web of
    plugins that register each other at import time.
    """

    @property
    def settings(self) -> Any: ...

    def build(self, handle: str, **params: Any) -> Any: ...

    def capability(self, name: str) -> Any: ...


@runtime_checkable
class PluginContextLike(Protocol):
    """The context handed to ``build``."""

    @property
    def settings(self) -> Any: ...

    def build(self, handle: str, **params: Any) -> Any: ...

    def capability(self, name: str) -> Any: ...


Builder = Callable[..., Any]
"""``build(ctx, **params)`` — the factory a plugin exposes."""


def format_handle(kind: PluginKind | str, name: str) -> str:
    kind_value = kind.value if isinstance(kind, PluginKind) else str(kind)
    return f"{kind_value}:{name}"


def parse_handle(handle: str) -> tuple[str, str]:
    """Split ``"source:upstox"`` into ``("source", "upstox")``.

    A bare name is rejected rather than guessed at: resolving an ambiguous
    handle by searching every kind is exactly the kind of convenience that makes
    a plugin system impossible to reason about later.
    """
    kind, separator, name = handle.partition(":")
    if not separator or not kind or not name:
        raise ValueError(f"malformed plugin handle {handle!r}; expected 'kind:name'")
    return kind, name


def iter_capabilities(manifests: Iterator[PluginManifest]) -> Iterator[tuple[str, str]]:
    for manifest in manifests:
        for capability in manifest.provides:
            yield capability, manifest.handle
