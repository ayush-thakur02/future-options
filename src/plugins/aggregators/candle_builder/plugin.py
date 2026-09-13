"""Tick-to-candle aggregation as a plugin.

Provides the ``bars`` capability, which is what every downstream stage actually
depends on. Swapping this pack for one that builds tick bars or volume bars
changes nothing above it.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .aggregator import CandleAggregator, bar_start

MANIFEST = PluginManifest(
    name="candle_builder",
    kind=PluginKind.AGGREGATOR,
    description="Session-anchored time bars with a tick-count activity proxy",
    provides=("bars",),
    tags=("timebars",),
)


def build(ctx: PluginContext, bar_minutes: int | None = None, **params) -> CandleAggregator:
    """Build an aggregator. Bar size defaults to the configured timeframe."""
    return CandleAggregator(bar_minutes=bar_minutes or ctx.settings.bar_minutes)


__all__ = ["MANIFEST", "CandleAggregator", "bar_start", "build"]
