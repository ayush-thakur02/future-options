"""Errors raised by the plugin kernel.

Kept together and narrow so a caller can distinguish "you asked for something
that does not exist" (``UnknownPlugin``, ``NoProvider``) from "the plugin set you
composed is inconsistent" (``MissingDependency``), which need different
responses: the first is a typo, the second is a wiring mistake.
"""

from __future__ import annotations


class PluginError(RuntimeError):
    """Base class for every kernel failure."""


class PluginLoadError(PluginError):
    """A plugin module could not be imported or does not expose a valid plugin."""


class DuplicatePlugin(PluginError):
    """Two plugins were registered under the same handle, or claimed one capability."""


class UnknownPlugin(PluginError):
    """A handle was requested that is not registered."""


class NoProvider(PluginError):
    """A required capability has no registered provider."""


class MissingDependency(PluginError):
    """A composed plugin set does not satisfy every declared requirement."""
