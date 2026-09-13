"""The simulated market as a plugin.

A leaf source: it depends on nothing and nothing depends on it. The runtime picks
it when there is no Upstox token, or when ``--offline`` is passed, so the entire
pipeline — aggregator, features, strategies, forecasts, projected candles —
can be exercised end to end without an account.
"""

from __future__ import annotations

import pandas as pd

from core.settings import Settings
from kernel import PluginContext, PluginKind, PluginManifest

from .series import generate_candles
from .ticks import generate_ticks

MANIFEST = PluginManifest(
    name="simulated",
    kind=PluginKind.SOURCE,
    description="Generated NIFTY-like bars and ticks, replayable in real time",
    tags=("simulated", "offline", "replay"),
    params={"seed": 7, "days": 120},
)


class SimulatedMarket:
    """A generated market that behaves like a source.

    Holds the seed so a run is reproducible: the same seed and the same number of
    days produce the same series, which is what makes a bug found on generated
    data a bug that can be found again.
    """

    def __init__(self, settings: Settings, seed: int = 7, days: int = 120) -> None:
        self.settings = settings
        self.seed = int(seed)
        self.days = int(days)

    def candles(self, days: int | None = None, bar_minutes: int | None = None) -> pd.DataFrame:
        return generate_candles(
            days=days or self.days,
            bar_minutes=bar_minutes or self.settings.bar_minutes,
            seed=self.seed,
        )

    def ticks(self, bars: pd.DataFrame, ticks_per_bar: int = 6) -> pd.DataFrame:
        """Expand bars into a tick stream, seeded from the bars themselves."""
        return generate_ticks(bars, ticks_per_bar=ticks_per_bar, seed=self.seed + 4)

    def __repr__(self) -> str:
        return f"<SimulatedMarket seed={self.seed} days={self.days}>"


def build(ctx: PluginContext, seed: int = 7, days: int = 120, **params) -> SimulatedMarket:
    return SimulatedMarket(settings=ctx.settings, seed=seed, days=days)
