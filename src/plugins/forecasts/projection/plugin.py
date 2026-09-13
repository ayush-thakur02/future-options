"""The projection pack as a plugin.

``bars_ahead`` defaults to three — the next three candles — and is tunable from
``plugins: {"forecast:projection": {bars_ahead: 5}}`` in ``config/default.yaml``
without touching code, which is the point of declaring it in the manifest.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from .candles import DEFAULT_BARS_AHEAD
from .forecaster import ProjectionForecaster

MANIFEST = PluginManifest(
    name="projection",
    kind=PluginKind.FORECAST,
    description="The next three candles, re-projected every second from the live price",
    provides=("projection",),
    tags=("nowcast", "live", "candles"),
    params={"bars_ahead": DEFAULT_BARS_AHEAD},
)


def build(ctx: PluginContext, bars_ahead: int = DEFAULT_BARS_AHEAD, **params) -> ProjectionForecaster:
    return ProjectionForecaster(bars_ahead=bars_ahead, logger=ctx.logger)
