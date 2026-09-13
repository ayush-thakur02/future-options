"""The context a plugin's ``build`` receives.

Everything a plugin legitimately needs at construction time: the resolved
settings, the event bus, and a way to build other plugins. It is the only object
that crosses from the kernel into plugin code, so the plugin-facing surface of
the kernel is exactly this class plus the manifest type.

Handed to ``build(ctx, **params)``, never stored globally, so two kernels can
exist in one process — which is what lets the tests build a throwaway kernel
without disturbing a live one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.settings import Settings
from .bus import EventBus
from .registry import Registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kernel import Kernel


@dataclass(slots=True)
class PluginContext:
    """What a plugin is given when it is built."""

    _kernel: Kernel
    bus: EventBus
    registry: Registry
    params: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("niftypulse"))

    @property
    def settings(self) -> Settings:
        return self._kernel.settings

    def build(self, handle: str, **params: Any) -> Any:
        """Build another plugin by handle, so a plugin can compose its own parts."""
        return self._kernel.build(handle, **params)

    def capability(self, name: str) -> Any:
        """Build whichever plugin provides ``name``."""
        return self._kernel.capability(name)

    def handle_for(self, capability: str) -> str:
        """Resolve a capability to the handle that provides it, without building it."""
        return self.registry.provider(capability)

    def has_capability(self, name: str) -> bool:
        return name in self.registry.capabilities()


__all__ = ["PluginContext"]
