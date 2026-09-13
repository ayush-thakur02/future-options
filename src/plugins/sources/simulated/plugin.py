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
from plugins.sources import StatusHandler, TickHandler

from .feed import DEFAULT_TICKS_PER_BAR, SimulatedFeed
from .series import generate_candles, rebase
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

    def candles(
        self,
        days: int | None = None,
        bar_minutes: int | None = None,
        rebase_to: float | None = None,
    ) -> pd.DataFrame:
        """Generated bars, optionally scaled to open at ``rebase_to``.

        The rebase is what stops a replay from jumping at its first tick when it
        continues real history: the shape is generated, but the level is the
        market's.
        """
        frame = generate_candles(
            days=days or self.days,
            bar_minutes=bar_minutes or self.settings.bar_minutes,
            seed=self.seed,
        )
        return rebase(frame, rebase_to) if rebase_to else frame

    def ticks(self, bars: pd.DataFrame, ticks_per_bar: int = 6) -> pd.DataFrame:
        """Expand bars into a tick frame, for headless replay."""
        return generate_ticks(bars, ticks_per_bar=ticks_per_bar, seed=self.seed + 4)

    def feed(
        self,
        bars: pd.DataFrame,
        on_tick: TickHandler,
        on_status: StatusHandler | None = None,
        speed: float = 1.0,
        ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
        bar_minutes: int | None = None,
        loop: bool = True,
        close_prev: float = 0.0,
    ) -> SimulatedFeed:
        """A tick stream paced by the wall clock, for the live dashboard.

        Paced rather than fast: a replay that finishes a trading day in two
        seconds gives the projected candles no seconds to move in, which is the
        one thing they exist to do.
        """
        return SimulatedFeed(
            bars=bars,
            on_tick=on_tick,
            on_status=on_status,
            speed=speed,
            ticks_per_bar=ticks_per_bar,
            bar_minutes=bar_minutes or self.settings.bar_minutes,
            loop=loop,
            close_prev=close_prev,
        )

    def __repr__(self) -> str:
        return f"<SimulatedMarket seed={self.seed} days={self.days}>"


def build(ctx: PluginContext, seed: int = 7, days: int = 120, **params) -> SimulatedMarket:
    return SimulatedMarket(settings=ctx.settings, seed=seed, days=days)
