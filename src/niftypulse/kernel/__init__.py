"""The plugin kernel.

*Everything is a plugin* here in a specific, checkable sense: a capability is
added to the platform by dropping a folder into the plugin tree, not by editing
a registry, a factory, or an if-statement. The kernel knows nothing about market
data, indicators, or candlesticks — it knows how to find plugins, build them,
and connect them through declared capabilities.

    kernel = Kernel.bootstrap(settings)
    kernel.require(["source:simulated", "renderer:terminal"])
    renderer = kernel.build("renderer:terminal")

The pieces:

* :mod:`~niftypulse.kernel.contracts` — what a plugin is (manifest + build)
* :mod:`~niftypulse.kernel.registry` — handles, kinds, capability index
* :mod:`~niftypulse.kernel.loader` — discovery of bundled and installed plugins
* :mod:`~niftypulse.kernel.bus` — the topic bus plugins talk over
* :mod:`~niftypulse.kernel.context` — what a plugin is handed when built
* :mod:`~niftypulse.kernel.kernel` — the composition root
"""

from .bus import EventBus, Subscription, Topic
from .context import PluginContext
from .contracts import PluginKind, PluginManifest, format_handle, parse_handle
from .errors import (
    DuplicatePlugin,
    MissingDependency,
    NoProvider,
    PluginError,
    PluginLoadError,
    UnknownPlugin,
)
from .kernel import Kernel, bootstrap
from .loader import BUILTIN_PACKAGE, ENTRY_POINT_GROUP, LoadReport, discover, discover_entry_points
from .registry import PluginEntry, Registry

__all__ = [
    "BUILTIN_PACKAGE",
    "ENTRY_POINT_GROUP",
    "DuplicatePlugin",
    "EventBus",
    "Kernel",
    "LoadReport",
    "MissingDependency",
    "NoProvider",
    "PluginContext",
    "PluginEntry",
    "PluginError",
    "PluginKind",
    "PluginLoadError",
    "PluginManifest",
    "Registry",
    "Subscription",
    "Topic",
    "UnknownPlugin",
    "bootstrap",
    "discover",
    "discover_entry_points",
    "format_handle",
    "parse_handle",
]
