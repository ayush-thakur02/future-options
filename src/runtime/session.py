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
from .warmup import WarmUp, WarmUpReport, warm_up_checkpoint

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
    research: bool = True
    # Replay recent bars into the research ledgers before the session starts, so
    # the strategies and the online learners are not beginning the day with no
    # evidence behind them.
    warmup: bool = True
    # Bars a cold instrument is replayed over. Capped at the engine's own strategy
    # window, because that is all a rule is ever evaluated on live.
    warmup_bars: int = 600
    warmup_projections: bool = True
    # Browser bind overrides. None keeps the value configured for the renderer.
    web_host: str | None = None
    web_port: int | None = None
    open_browser: bool | None = None


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
    warmup: WarmUpReport | None = field(init=False, default=None)
    live: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        settings = self.kernel.settings
        if self.config.timeframe:
            settings.bar_minutes = self.config.timeframe

        self.broker = self.kernel.build("source:upstox")
        self.live = not self.config.offline and bool(getattr(self.broker, "is_configured", False))
        self.feed_status = "initialising"
        self.tick_recorders = {}
        self._tick_queue = None
        self._fatal_error = None

        self.engine = Engine(
            self.kernel,
            bar_minutes=settings.bar_minutes,
            calendar=TradingCalendar(),
            source="Upstox" if self.live else "simulation",
        )
        self.renderer = self._build_renderer()
        self.loader = BarLoader(self.kernel, logger=self.logger)

    # ------------------------------------------------------------- composition

    def _build_renderer(self):
        if "web_frame" not in self.kernel.registry.capabilities():
            return None
        overrides = {}
        if self.config.web_host is not None:
            overrides["host"] = self.config.web_host
        if self.config.web_port is not None:
            overrides["port"] = self.config.web_port
        if self.config.open_browser is not None:
            overrides["open_browser"] = self.config.open_browser
        return self.kernel.build("renderer:web", **overrides)

    def bootstrap(self, progress: Callable[[str], None] | None = None) -> Session:
        """Load history and warm everything up, so the first frame is readable.

        ``progress`` is called at each stage. Warming three instruments is real
        work — a feature matrix and the full strategy catalog per instrument — and a
        silent pause before the first frame reads as a hang.
        """
        report = progress or (lambda _message: None)
        started = time.perf_counter()

        if not self.config.offline:
            if not self.live:
                from plugins.sources import DataUnavailable
                raise DataUnavailable(
                    "Live dashboard requires a valid Upstox token. Authenticate with "
                    "plugins.sources.upstox.auth.interactive_login, or start with --offline."
                )
            report("checking Upstox authentication")
            self.broker.validate_auth()
            report(self.broker.auth_status)

        report("loading bars")
        if self.live:
            from core.calendar import IST
            from plugins.sources.history.resample import resample_ohlcv

            # The history plugin materializes the local WebSocket tape first and
            # asks REST only for coverage that is still absent.
            bars = self.loader.load(
                days=self.config.days,
                refresh=self.config.refresh_history,
                quiet=False,
                offline=False,
            )
            bars = resample_ohlcv(bars, self.kernel.settings.bar_minutes)
            cutoff = pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=self.kernel.settings.bar_minutes)
            bars = bars[bars.index <= cutoff]
            if bars.empty:
                raise RuntimeError("Upstox returned no historical bars; cannot warm a live research session.")
        else:
            bars = self.loader.load(days=self.config.days, refresh=False, quiet=False, offline=True)
            from plugins.sources.history.resample import resample_ohlcv

            bars = resample_ohlcv(bars, self.kernel.settings.bar_minutes)
        report(f"  {len(bars):,} bars in {time.perf_counter() - started:.1f}s")

        warm_started = time.perf_counter()
        if self.config.legs and self.kernel.registry.capabilities().get("option_chain"):
            report(f"warming instruments (index + legs) over {self.config.days} days")
            self.board = MarketBoard(
                self.kernel,
                bar_minutes=self.kernel.settings.bar_minutes,
                logger=self.logger,
                live=self.live,
                days=self.config.days,
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

        self._start_recording()
        if self.config.research:
            engines = [leg.engine for leg in self.board.legs] if self.board else [self.engine]
            labels = [leg.label for leg in self.board.legs] if self.board else ["INDEX"]
            for engine in engines:
                engine.enable_research()
            if self.config.warmup:
                self.warmup = self._run_warmup(list(zip(labels, engines, strict=True)), report)
        self.feed_status = "authenticated · awaiting ticks" if self.live else "SIMULATION"
        report(f"ready in {time.perf_counter() - started:.1f}s")
        return self

    def _run_warmup(self, targets, report: Callable[[str], None]) -> WarmUpReport:
        """Replay recent bars so the first live decision is not the first evidence."""
        service = WarmUp(
            checkpoint=warm_up_checkpoint(self.kernel.settings),
            logger=self.logger,
        )
        report(
            f"warming research (up to {self.config.warmup_bars:,} bars per instrument)"
        )
        outcome = service.run(
            targets,
            mode="live" if self.live else "simulation",
            bars=self.config.warmup_bars,
            projections=self.config.warmup_projections,
            progress=report,
        )
        report(f"  {outcome.headline()}")
        # The replay drove every forecaster through the past. Put them all back on
        # the live bar before anything is drawn from one.
        if self.board is not None:
            self.board.refresh_projection()
            self.board.warmup = outcome.as_row()
        else:
            self.engine.refresh_projection()
        return outcome

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
        keys = [leg.instrument_key for leg in self.board.legs] if self.board else [settings.instrument_key]
        for key in keys:
            self.tick_recorders[key] = TickRecorder(store=PartitionedStore(
                settings.data_dir, key, dataset=TICKS, shard=HOUR, normalizer=normalize_frames))
        self.tick_recorder = self.tick_recorders[settings.instrument_key]
        self.chain_recorder = ChainRecorder(
            store=PartitionedStore(
                settings.data_dir,
                f"{settings.instrument_key}-chain",
                dataset=CHAIN,
                shard=HOUR,
                normalizer=normalize_frames,
            )
        )
        self.logger.info("recording ticks and chain samples under %s", settings.data_dir)

    def _sample_chain(self) -> None:
        """Take a chain snapshot if enough time has passed since the last one."""
        if self.chain_recorder is None or self.source_chain is None:
            return
        now = time.monotonic()
        if not self.chain_recorder.due(now):
            return
        try:
            refresh = getattr(self.source_chain, "refresh_chain", None)
            if callable(refresh):
                refresh()
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
        strategies = f"{len(self.engine.signals)} research strategies ({len(self.engine.catalog)} available)"
        predictor = self.engine.predictor
        models = (
            f"{len(predictor.loaded_horizons)} horizons" if predictor is not None and predictor.is_ready else "none"
        )
        return (
            f"source {source} · {self.config.timeframe or self.kernel.settings.bar_minutes}m bars · "
            f"{strategies} · batch models {models} · {self.kernel.settings.worker_count} CPU workers{legs}"
        )

    # ------------------------------------------------------------------ drives

    def snapshot(self):
        """The frame to draw: a whole board when the legs are on, else one market."""
        return self.board.snapshot() if self.board is not None else self.engine.snapshot()

    async def run(self) -> None:
        """Stream data, refresh the projection, and render, until interrupted."""
        if self.renderer is None:
            raise RuntimeError("no renderer plugin provides the 'web_frame' capability")
        if self.engine.history.empty:
            raise RuntimeError("no bars loaded; check the source and the cache")

        self._tick_queue = asyncio.Queue(maxsize=20_000)
        feed = self._build_feed()
        # The clock is driven by whatever owns the projections: a board refreshes
        # every leg, an engine refreshes itself. Both expose the same two methods.
        clock_target = self.board if self.board is not None else self.engine
        nowcast = NowcastLoop(clock_target, interval=self.config.nowcast_interval, logger=self.logger)

        tasks = [asyncio.create_task(nowcast.run(), name="nowcast")]
        tasks.append(asyncio.create_task(self._consume_ticks(), name="instrument-work"))
        if self.chain_recorder is not None:
            tasks.append(asyncio.create_task(self._chain_loop(), name="chain-recording"))
        if feed is not None:
            tasks.append(asyncio.create_task(self._drive(feed), name="feed"))

        try:
            await self._render_loop()
        finally:
            nowcast.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.get_running_loop().shutdown_default_executor()
            self.stop_recording()
            if self.board is not None:
                self.board.close()
            else:
                self.engine.close()
            close = getattr(self.broker, "close", None)
            if callable(close):
                close()

    def stop_recording(self) -> None:
        """Flush whatever is buffered. Called on exit and safe to call twice."""
        for recorder in (*self.tick_recorders.values(), self.chain_recorder):
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
            self.feed_status = self.engine.status

    async def _chain_loop(self) -> None:
        while True:
            await asyncio.to_thread(self._sample_chain)
            await asyncio.sleep(10)

    def _build_feed(self):
        """The one line that separates a live run from a replay."""
        if self.live:
            self.engine.status = "connecting"
            return self.broker.feed(
                on_tick=self._on_tick,
                on_status=self._on_status,
                instrument_keys=[leg.instrument_key for leg in self.board.legs] if self.board else None,
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
        # Continue AFTER the warm-up; replaying timestamps already in history
        # would let the online model see those candles before forecasting them.
        calendar = self.engine.calendar
        moment = self.engine.history.index[-1].to_pydatetime()
        stamps = []
        from core.calendar import SESSION_CLOSE
        for _ in range(len(bars)):
            moment += pd.Timedelta(minutes=self.engine.bar_minutes)
            if moment.time() >= SESSION_CLOSE or not calendar.is_trading_day(moment.date()):
                moment = calendar.next_open(moment)
            stamps.append(moment)
        bars = bars.copy()
        bars.index = pd.DatetimeIndex(stamps, name="ts")
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
        """Keep the socket reader cheap; processing runs on instrument workers."""
        if self._tick_queue is not None:
            try:
                self._tick_queue.put_nowait(tick)
            except asyncio.QueueFull:
                self._fatal_error = "Tick queue full; restart to resync candles."
                raise RuntimeError("Tick processing overloaded: queue full; restart to resync candles.") from None
            return
        self._process_tick(tick)

    def _process_tick(self, tick) -> None:
        recorder = self.tick_recorders.get(tick.instrument_key or self.kernel.settings.instrument_key)
        if recorder is not None:
            recorder.record(tick)
        if self.board is not None:
            self.board.on_tick(tick)
        else:
            self.engine.on_tick(tick)

    async def _consume_ticks(self) -> None:
        while True:
            tick = await self._tick_queue.get()
            batch = [tick]
            while not self._tick_queue.empty() and len(batch) < 300:
                batch.append(self._tick_queue.get_nowait())
            grouped = {}
            for item in batch:
                grouped.setdefault(item.instrument_key, []).append(item)
            def consume(items):
                for item in items:
                    self._process_tick(item)
            try:
                if self.board is not None and self.live:
                    loop = asyncio.get_running_loop()
                    await asyncio.gather(*(loop.run_in_executor(self.board.executor, consume, items)
                                           for items in grouped.values()))
                else:
                    await asyncio.to_thread(consume, batch)
            except Exception as exc:
                self.feed_status = f"processing error: {type(exc).__name__}; restart to resync"
                self._fatal_error = self.feed_status
                self.logger.exception("tick processing stopped")
                raise

    def _on_status(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "connected":
            self.feed_status = "connected" if self.live else "SIMULATION"
        elif kind == "disconnected":
            self.feed_status = "reconnecting" if self.live else "replay finished"
        elif kind == "market_info":
            segments = payload.get("segments", {})
            self.feed_status = segments.get("NSE_INDEX", self.feed_status)
            engines = [leg.engine for leg in self.board.legs] if self.board else [self.engine]
            for engine in engines:
                engine.market_status = segments.get(engine.instrument_key.split("|")[0])
            if self.feed_status not in {"NORMAL_OPEN", "PRE_OPEN_START", "PRE_OPEN_END"}:
                self.feed_status = f"market closed · {self.feed_status}"
        elif kind in {"connection_error", "auth_error"}:
            self.feed_status = str(payload.get("error", "feed failed"))
        self.engine.status = self.feed_status

    async def _render_loop(self) -> None:
        """Redraw at the renderer's cadence; the projection keeps its own.

        Inside the renderer's live block, which is what serves the dashboard. A
        renderer publishes nothing outside one — the socket is bound on entry and
        released on exit — so a loop that pushed frames without it wrote to a
        server nobody could reach while the process looked perfectly healthy.
        """
        renderer = self.renderer
        interval = max(float(self.config.refresh), 0.05)
        with renderer.live():
            try:
                while True:
                    if self._fatal_error:
                        raise RuntimeError(self._fatal_error)
                    frame = await asyncio.to_thread(self.snapshot)
                    renderer.live_update(frame, self.feed_status)
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
