"""The Upstox pack as a plugin.

Declares the ``broker`` capability: the singular, account-backed way to reach
Upstox data. The history store consumes it, which is the only link between the
two packs and is expressed as a capability rather than an import.

Tags are how the runtime picks a *source*: ``live`` marks this one, ``simulated``
the other, and ``niftypulse plugins --tag live`` finds it.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .broker import UpstoxBroker

MANIFEST = PluginManifest(
    name="upstox",
    kind=PluginKind.SOURCE,
    description="Live Upstox v3 WebSocket feed plus REST history and quotes",
    provides=("broker",),
    tags=("live", "broker", "upstox"),
    params={"feed_mode": "full"},
)


def build(ctx: PluginContext, feed_mode: str = "full", **params) -> UpstoxBroker:
    """Build the broker. Holds no token until one is needed, so this never fails
    on a machine without credentials — it only fails when live data is asked for.
    """
    return UpstoxBroker(settings=ctx.settings, feed_mode=feed_mode)
