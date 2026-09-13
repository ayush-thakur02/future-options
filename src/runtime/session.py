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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from core.calendar import TradingCalendar
from core.settings import Settings
from core.types import MarketSnapshot
from kernel import Kernel

from .bars import BarLoader
from .board import MarketBoard
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
    # Chart the call and put legs beside the index when a chain is available.
    legs: bool = True
    # Write the live tape and chain samples to the store while the session runs.
    # Only ever true for a live run: recording generated ticks into the real
    # store would mix invented prices into the only dataset that cannot be
    # re-fetched.
    record: bool = True


@dataclass
class Session:
    """A composed run: kernel, engine, source and renderer."""

    kernel: Kernel
    config: SessionConfig = field(default_factory=SessionConfig)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("niftypulse.session"))

    engine: Engine = field(init=False)
    board: MarketBoard | None = field(init=False, default=None)
    tick_recorder: object | None = field(init=False, default=None)
    chain_recorder: object | None = field(init=False, default=None)
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

    def bootstrap(self, progress: Callable[[str], None] | None = None) -> Session:
        """Load history and warm everything up, so the first frame is readable.

        ``progress`` is called at each stage. Warming three instruments is real
        work — a feature matrix and nineteen strategies per instrument — and a
        silent pause before the first frame reads as a hang.
        """
        report = progress or (lambda _message: None)
        started = time.perf_counter()

        report("loading bars")
        bars = self.loader.load(
            days=self.config.days,
            refresh=self.config.refresh_history and self.live,
            quiet=False,
            offline=self.config.offline,
        )
        report(f"  {len(bars):,} bars in {time.perf_counter() - started:.1f}s")

        warm_started = time.perf_counter()
        if self.config.legs and self.kernel.registry.capabilities().get("option_chain"):
            report(f"warming instruments (index + legs) over {self.config.days} days")
            self.board = MarketBoard(
                self.kernel,
                bar_minutes=self.kernel.settings.bar_minutes,
                logger=self.logger,
            ).build(bars)
            # The index engine of the board *is* this session's engine, so every
            # other part of the runtime — the feed, the clock, the renderer —
            # keeps talking to one thing.
            self.engine = self.board.index_engine
        else:
            report("warming the engine")
            self.engine.bootstrap(history=bars)
            self.engine.refresh_projection()

        report(f"  {len(self.engine.history):,} bars warmed in {time.perf_counter() - warm_started:.1f}s")

        if self.renderer is not None:
            self.renderer.forecaster = self.engine.forecaster

        self._start_recording()
        report(f"ready in {time.perf_counter() - started:.1f}s")
        return self

    def _start_recording(self) -> None:
        """Begin writing the tape and the chain, if this is a live run."""
        if not self.config.record or not self.live:
            return

        from plugins.sources.history import (
            CHAIN,
            HOUR,
            TICKS,
            ChainRecorder,
            PartitionedStore,
            TickRecorder,
            normalize_frames,
        )

        settings = self.kernel.settings
        self.tick_recorder = TickRecorder(
            store=PartitionedStore(
                settings.data_dir,
                settings.instrument_key,
                dataset=TICKS,
                shard=HOUR,
                normalizer=normalize_frames,
            )
        )
        self.chain_recorder = ChainRecorder(
            store=PartitionedStore(
                settings.data_dir,
                f"{settings.instrument_key}-chain",
                dataset=CHAIN,
                shard=HOUR,
                normalizer=normalize_frames,
            )
        )
        if self.board is not None:
            self.board.on_refresh = self._sample_chain
        self.logger.info("recording ticks and chain samples under %s", settings.data_dir)

    def _sample_chain(self) -> None:
        """Take a chain snapshot if enough time has passed since the last one."""
        if self.chain_recorder is None or self.source_chain is None:
            return
        now = time.monotonic()
        if not self.chain_recorder.due(now):
            return
        try:
            chain = self.source_chain.chain(self.engine.last_price, bars=self.engine.market_bars())
        except Exception as exc:  # noqa: BLE001 — a failed sample is not a failed session
            self.logger.debug("chain sample failed: %s", exc)
            return
        self.chain_recorder.record(chain, now)

    @property
    def source_chain(self):
        return getattr(self.board, "source", None)

    def describe(self) -> str:
        source = "upstox (live)" if self.live else "simulated"
        legs = f" · {self.board.describe()}" if self.board is not None else ""
        strategies = self.engine.catalog.summary()
        predictor = self.engine.predictor
        models = (
            f"{len(predictor.loaded_horizons)} horizons" if predictor is not None and predictor.is_ready else "none"
        )
        return (
            f"source {source} · {self.config.timeframe or self.kernel.settings.bar_minutes}m bars · "
            f"{strategies} · models {models}{legs}"
        )

    # ------------------------------------------------------------------ drives

    def snapshot(self):
        """The frame to draw: a whole board when the legs are on, else one market."""
        return self.board.snapshot() if self.board is not None else self.engine.snapshot()

    async def run(self) -> None:
        """Stream data, refresh the projection, and render, until interrupted."""
        if self.renderer is None:
            raise RuntimeError("no renderer plugin provides the 'frame' capability")
        if self.engine.history.empty:
            raise RuntimeError("no bars loaded; check the source and the cache")

        feed = self._build_feed()
        # The clock is driven by whatever owns the projections: a board refreshes
        # every leg, an engine refreshes itself. Both expose the same two methods.
        clock_target = self.board if self.board is not None else self.engine
        nowcast = NowcastLoop(clock_target, interval=self.config.nowcast_interval, logger=self.logger)

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
            self.stop_recording()
            close = getattr(self.broker, "close", None)
            if callable(close):
                close()

    def stop_recording(self) -> None:
        """Flush whatever is buffered. Called on exit and safe to call twice."""
        for recorder in (self.tick_recorder, self.chain_recorder):
            flush = getattr(recorder, "close", None)
            if callable(flush):
                try:
                    flush()
                except Exception as exc:  # noqa: BLE001 — never lose the session to a flush
                    self.logger.warning("failed to flush the store: %s", exc)

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
                on_tick=self._on_tick,
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
            on_tick=self._on_tick,
            on_status=self._on_status,
            speed=self.config.speed,
            ticks_per_bar=self.config.ticks_per_bar,
            bar_minutes=self.kernel.settings.bar_minutes,
            close_prev=self.engine.prev_close,
        )

    def _on_tick(self, tick) -> None:
        """One index tick in: record it, then let the board derive the legs."""
        if self.tick_recorder is not None:
            self.tick_recorder.record(tick)
        if self.board is not None:
            self.board.on_tick(tick)
        else:
            self.engine.on_tick(tick)

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
