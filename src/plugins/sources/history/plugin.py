"""The history pack as a plugin.

Provides the ``history`` capability — the one place bars are persisted — and
consumes ``broker`` for the bars it cannot find on disk. The requirement is
declared rather than checked by import, so a build with no broker registered
fails loudly at composition time instead of quietly at 09:20 on a trading day.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .service import HistorySource

MANIFEST = PluginManifest(
    name="history",
    kind=PluginKind.SOURCE,
    description="Local parquet candle cache, topped up from the broker",
    provides=("history",),
    requires=("broker",),
    tags=("history", "cache", "offline"),
    params={"offline": False},
)


def build(ctx: PluginContext, offline: bool = False, **params) -> HistorySource:
    """Build the cache. The broker is optional at runtime: without a token the
    cache simply serves whatever it already holds, or generated bars offline.
    """
    broker = ctx.capability("broker") if ctx.has_capability("broker") else None
    return HistorySource(settings=ctx.settings, broker=broker, offline=offline)
