"""Composing a run: which plugins, in what order, driven how.

This is the only place that knows the platform is a dashboard watching NIFTY. The
kernel knows how to find and build plugins; the packs know how to do their jobs;
the session is what decides that a run needs a source, bars, an engine, a
projection and a renderer, and that they are driven at two cadences.

Live and simulated differ in exactly one line — which feed is built — which is
the point of having pushed the source decision down to a plugin.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pandas as pd

from core.calendar import TradingCalendar
from core.settings import Settings
from core.types import MarketSnapshot
from kernel import Kernel

from .bars import BarLoader
from .engine import Engine
from .nowcast import NowcastLoop

DEFAULT_DAYS = 15
DEFAULT_TICKS_PER_BAR = 12


@dataclass
class SessionConfig:
    """How this run should behave."""

    offline: bool = False
    timeframe: int | None = None
    refresh: float = 1.0
    nowcast_interval: float = 1.0
    days: int = DEFAULT_DAYS
    speed: float = 1.0
    ticks_per_bar: int = DEFAULT_TICKS_PER_BAR
    bars_ahead: int | None = None
    refresh_history: bool = True
    # Bars replayed by Session.replay when no series is supplied.
    replay_bars: int = 40


@dataclass
class Session:
    """A composed run: kernel, engine, source and renderer."""

    kernel: Kernel
    config: SessionConfig = field(default_factory=SessionConfig)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("niftypulse.session"))

    engine: Engine = field(init=False)
    renderer: object | None = field(init=False, default=None)
    live: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        settings = self.kernel.settings
        if self.config.timeframe:
            settings.bar_minutes = self.config.timeframe

        self.broker = self.kernel.build("source:upstox")
        self.live = not self.config.offline and bool(getattr(self.broker, "is_configured", False))

        self.engine = Engine(
            self.kernel,
            bar_minutes=settings.bar_minutes,
            calendar=TradingCalendar(),
        )
        self.renderer = self._build_renderer()
        self.loader = BarLoader(self.kernel, logger=self.logger)

    # ------------------------------------------------------------- composition

    def _build_renderer(self):
        if "frame" not in self.kernel.registry.capabilities():
            return None
        return self.kernel.build(
            "renderer:terminal",
            refresh=self.config.refresh,
            forecaster=self.engine.forecaster,
        )

    def bootstrap(self) -> Session:
        """Load history and warm the engine up, so the first frame is readable."""
        bars = self.loader.load(
            days=self.config.days,
            refresh=self.config.refresh_history and self.live,
            quiet=False,
            offline=self.config.offline,
        )
        self.engine.bootstrap(history=bars)
        self.engine.refresh_projection()
        return self

    def describe(self) -> str:
        source = "upstox (live)" if self.live else "simulated"
        strategies = self.engine.catalog.summary()
        predictor = self.engine.predictor
        models = (
            f"{len(predictor.loaded_horizons)} horizons" if predictor is not None and predictor.is_ready else "none"
        )
        return (
            f"source {source} · {self.config.timeframe or self.kernel.settings.bar_minutes}m bars · "
            f"{strategies} · models {models}"
        )

    # ------------------------------------------------------------------ drives

    def snapshot(self) -> MarketSnapshot:
        return self.engine.snapshot()

    async def run(self) -> None:
        """Stream data, refresh the projection, and render, until interrupted."""
        if self.renderer is None:
            raise RuntimeError("no renderer plugin provides the 'frame' capability")
        if self.engine.history.empty:
            raise RuntimeError("no bars loaded; check the source and the cache")

        feed = self._build_feed()
        nowcast = NowcastLoop(self.engine, interval=self.config.nowcast_interval, logger=self.logger)

        tasks = [asyncio.create_task(nowcast.run(), name="nowcast")]
        if feed is not None:
            tasks.append(asyncio.create_task(self._drive(feed), name="feed"))

        try:
            await self._render_loop()
        finally:
            nowcast.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            close = getattr(self.broker, "close", None)
            if callable(close):
                close()

    async def _drive(self, feed) -> None:
        """Run the feed, reporting a failure as status rather than as a crash."""
        try:
            await feed.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — one dead feed must not kill the UI
            self.logger.warning("feed stopped: %s", exc)
            self.engine.status = f"feed error: {str(exc)[:60]}"

    def _build_feed(self):
        """The one line that separates a live run from a replay."""
        if self.live:
            self.engine.status = "connecting"
            return self.broker.feed(
                on_tick=self.engine.on_tick,
                on_status=self._on_status,
            )

        market = self.kernel.build("source:simulated", days=self.config.days)
        # Re-based onto the price the engine is actually at, so the replay
        # continues the history on screen instead of opening 1,800 points away
        # from it — which is what a generated series does when it starts from its
        # own base level.
        bars = market.candles(
            days=max(self.config.days, 2),
            bar_minutes=self.kernel.settings.bar_minutes,
            rebase_to=self.engine.last_price or None,
        )
        self.engine.status = "replaying"
        return market.feed(
            bars=bars,
            on_tick=self.engine.on_tick,
            on_status=self._on_status,
            speed=self.config.speed,
            ticks_per_bar=self.config.ticks_per_bar,
            bar_minutes=self.kernel.settings.bar_minutes,
            close_prev=self.engine.prev_close,
        )

    def _on_status(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "connected":
            self.engine.status = "connected" if self.live else "replay"
        elif kind == "disconnected":
            self.engine.status = "reconnecting" if self.live else "replay finished"
        elif kind == "market_info":
            segments = payload.get("segments", {})
            self.engine.status = segments.get("NSE_INDEX", self.engine.status)
        elif kind == "connection_error":
            self.engine.status = f"feed error: {str(payload.get('error', ''))[:40]}"

    async def _render_loop(self) -> None:
        """Redraw at the renderer's cadence; the projection keeps its own."""
        renderer = self.renderer
        interval = max(float(self.config.refresh), 0.05)
        try:
            while True:
                renderer.live_update(self.engine.snapshot(), self.engine.status)
                await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

    # ------------------------------------------------------------------ replay

    def replay(self, bars: pd.DataFrame | None = None) -> list[MarketSnapshot]:
        """Run the whole thing headlessly and collect the frames at bar closes.

        Used by tests and by the offline snapshot: the same engine, the same
        projections, no clock. The projection is refreshed on bar close here
        rather than every second — there are no seconds to spend.
        """
        market = self.kernel.build("source:simulated", days=self.config.days)
        if bars is None:
            # The engine's own recent bars, so the replay continues the history
            # it is drawn against rather than arriving from somewhere else.
            bars = self.engine.history.tail(self.config.replay_bars)
        ticks = market.ticks(bars, ticks_per_bar=self.config.ticks_per_bar)
        return list(self.engine.replay(ticks))


def build_session(
    settings: Settings | None = None,
    config: SessionConfig | None = None,
    logger: logging.Logger | None = None,
) -> Session:
    """Bootstrap a kernel and compose a session from it."""
    settings = settings or Settings()
    kernel = Kernel.bootstrap(settings)
    return Session(kernel=kernel, config=config or SessionConfig(), logger=logger or logging.getLogger("niftypulse.session"))


__all__ = ["Session", "SessionConfig", "build_session"]
