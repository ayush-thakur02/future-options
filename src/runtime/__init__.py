"""The runtime: where the plugin graph is composed and driven.

The kernel knows how to find and build plugins. The runtime knows *which* ones
this application wants, and how they are driven:

* :mod:`~runtime.bars` — the policy for where the bar series comes from
* :mod:`~runtime.engine` — the live state machine, per tick and per bar
* :mod:`~runtime.nowcast` — the per-second projection refresh

Nothing here is a plugin. Keeping composition out of the plugin tree is what
lets any pack be replaced without touching the application that uses it.
"""

from .bars import BarLoader

__all__ = ["BarLoader"]
