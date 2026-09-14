"""Tests for the runtime: conviction, the engine, the refresh clock, the feed.

These cover the behaviour the live dashboard actually depends on — that a bar
close scores the projections that targeted it, that the refresh clock runs on its
own schedule rather than on ticks, and that the simulated feed is paced by the
wall clock instead of running as fast as the CPU allows.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.calendar import IST
from core.settings import Settings
from core.types import Tick
from kernel import Kernel, PluginEntry, PluginKind, PluginManifest
from plugins.sources.simulated import SimulatedFeed, SimulatedMarket
from plugins.sources.simulated.feed import _intra_bar_path
from runtime.conviction import blend, trend_conviction
from runtime.engine import Engine
from runtime.nowcast import NowcastLoop
from runtime.session import Session, SessionConfig


@pytest.fixture(scope="module")
def market() -> SimulatedMarket:
    return SimulatedMarket(Settings(), seed=5, days=3)


@pytest.fixture(scope="module")
def bars(market) -> pd.DataFrame:
    return market.candles(days=3)


def kernel_for(settings: Settings | None = None) -> Kernel:
    return Kernel.bootstrap(settings or Settings())


def engine_for(bars, settings: Settings | None = None, **kwargs) -> Engine:
    engine = Engine(kernel_for(settings), **kwargs)
    engine.bootstrap(history=bars)
    return engine


@pytest.fixture(scope="module")
def warmed(bars) -> Engine:
    """One warmed engine for the read-only assertions.

    Warming an engine means computing a full feature matrix and scoring thirty
    strategies over the signal window, which is seconds of work. Sharing it keeps
    the suite honest about what it is testing — the engine's behaviour, not how
    fast it can be rebuilt.
    """
    engine = engine_for(bars.tail(400))
    engine.refresh_projection()
    return engine


# ---------------------------------------------------------------- conviction


def ramp(step: float, bars: int = 60) -> pd.DataFrame:
    """A clean trend whose high/low track the close, so the ATR stays meaningful."""
    closes = [100.0 + index * step for index in range(bars)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [price + 1.0 for price in closes],
            "low": [price - 1.0 for price in closes],
            "close": closes,
        }
    )


def test_trend_conviction_is_positive_on_a_rising_series() -> None:
    assert trend_conviction(ramp(0.5)) > 0.5


def test_trend_conviction_is_negative_on_a_falling_series() -> None:
    assert trend_conviction(ramp(-0.5)) < -0.5


def test_trend_conviction_is_bounded_and_quiet_on_a_flat_series() -> None:
    flat = pd.DataFrame(
        {"open": [100.0] * 60, "high": [100.0] * 60, "low": [100.0] * 60, "close": [100.0] * 60}
    )
    value = trend_conviction(flat)
    assert -1.0 <= value <= 1.0
    assert abs(value) < 0.05


def test_trend_conviction_handles_a_tiny_series() -> None:
    """The first seconds of a replay have almost no bars; it must not raise."""
    tiny = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]})
    assert trend_conviction(tiny) == 0.0
    assert trend_conviction(pd.DataFrame()) == 0.0


def test_blend_weights_the_fast_read_and_clamps() -> None:
    assert blend(0.0, 1.0, fast_weight=0.5) == pytest.approx(0.5)
    assert blend(1.0, 1.0) == pytest.approx(1.0)
    assert blend(-1.0, -1.0) == pytest.approx(-1.0)
    assert -1.0 <= blend(5.0, 5.0) <= 1.0
    assert blend(0.8, 0.0, fast_weight=0.0) == pytest.approx(0.8)


# -------------------------------------------------------------------- engine


def test_engine_takes_everything_from_the_kernel(warmed) -> None:
    assert warmed.catalog.names()
    assert warmed.forecaster is not None
    assert warmed.kernel.registry.capabilities()["features"] == "features:technical"


def test_engine_projects_the_next_three_candles(warmed) -> None:
    assert warmed.refresh_projection()
    assert len(warmed.projections) == 3
    assert [candle.horizon for candle in warmed.projections] == [1, 2, 3]


def test_projection_is_stamped_after_the_last_bar(warmed) -> None:
    """Times come from the data's clock, so a cached snapshot cannot contradict itself."""
    last_bar = warmed.market_bars().index[-1]
    assert warmed.projections[0].ts > last_bar.to_pydatetime()


def test_snapshot_carries_the_projection(warmed) -> None:
    snapshot = warmed.snapshot()
    assert len(snapshot.projections) == 3
    assert snapshot.conviction == warmed.conviction
    assert snapshot.projection_ts is not None


def test_engine_without_a_projection_plugin_reports_nothing(bars) -> None:
    """The projection is optional: a build without it must still run."""
    kernel = Kernel(Settings())
    for entry in Kernel.bootstrap(Settings()).registry.find():
        if entry.kind is not PluginKind.FORECAST:
            kernel.register(entry)
    engine = Engine(kernel)
    engine.bootstrap(history=bars)
    assert engine.refresh_projection() is False
    assert engine.snapshot().projections == []


def test_engine_needs_enough_bars_before_projecting() -> None:
    engine = Engine(kernel_for())
    assert engine.refresh_projection() is False


def test_market_bars_include_the_forming_bar() -> None:
    """A projection taken mid-bar must start from the price, not the last close."""
    engine = Engine(kernel_for())
    start = datetime(2026, 3, 10, 11, 0, tzinfo=IST)
    for step in range(5):
        engine.on_tick(Tick(ts=start + timedelta(seconds=step * 5), ltp=25_000.0 + step))

    merged = engine.market_bars()
    assert len(merged) == 1, "one forming bar, no history"
    assert float(merged["close"].iloc[-1]) == pytest.approx(25_004.0)


def test_tick_updates_the_anchor_without_recomputing() -> None:
    engine = Engine(kernel_for())
    engine.on_tick(Tick(ts=datetime(2026, 3, 10, 11, 0, tzinfo=IST), ltp=24_123.0))
    assert engine.last_price == pytest.approx(24_123.0)
    assert engine.history.empty, "a tick inside a bar must not close or rewrite history"


def test_bar_close_refreshes_the_projection_and_scores_it(warmed) -> None:
    """The whole point: what was projected gets judged against what printed.

    Four bars are walked so the projections taken one, two and three bars ahead
    each get a bar of their own to be scored against.
    """
    engine = warmed
    before = engine.forecaster.tracker.overall().scored
    start = engine.market_bars().index[-1].to_pydatetime() + timedelta(minutes=1)

    for minute in range(4):
        for second in (0, 30):
            moment = start + timedelta(minutes=minute, seconds=second)
            engine.on_tick(Tick(ts=moment, ltp=24_400.0 + minute * 4 + second / 60))

    after = engine.forecaster.tracker.overall().scored
    assert after > before, "a closed bar must score the projections aimed at it"
    first = engine.forecaster.tracker.stat(1)
    assert first is not None and first.scored >= 1
    assert 0.0 <= first.hit_rate <= 1.0
    assert first.mean_abs_error_bps >= 0.0


def test_engine_conviction_is_bounded(warmed) -> None:
    warmed.slow_conviction = 5.0
    warmed.refresh_projection()
    assert -1.0 <= warmed.conviction <= 1.0
    warmed.slow_conviction = 0.0


def test_engine_replay_yields_a_frame_per_bar(bars, market) -> None:
    engine = engine_for(bars.tail(200))
    ticks = market.ticks(bars.tail(24), ticks_per_bar=3)
    frames = list(engine.replay(ticks))
    assert frames
    assert all(frame.candles is not None for frame in frames)


# ------------------------------------------------------------------- nowcast


def test_nowcast_refreshes_and_publishes(warmed) -> None:
    engine = warmed
    seen: list[object] = []
    engine.subscribe(seen.append)

    loop = NowcastLoop(engine, interval=0.01)
    assert loop.tick_once() is True
    assert loop.stats.refreshes == 1
    assert loop.stats.skipped == 0
    assert len(seen) == 1, "listeners must see the refreshed frame"
    assert seen[0].projections


def test_nowcast_survives_a_failing_refresh() -> None:
    """A projection is not worth taking the dashboard down for."""

    class Broken(Engine):
        def refresh_projection(self) -> bool:
            raise RuntimeError("boom")

    loop = NowcastLoop(Broken(kernel_for()), interval=0.01)
    assert loop.tick_once() is False
    assert loop.stats.skipped == 1


def test_nowcast_counts_a_skipped_refresh() -> None:
    loop = NowcastLoop(Engine(kernel_for()), interval=0.01)
    assert loop.tick_once() is False
    assert loop.stats.skipped == 1
    assert "refreshes" in loop.stats.summary


async def test_nowcast_loop_runs_on_its_own_clock(warmed) -> None:
    """It must refresh even when no tick arrives, which is the quiet-tape case."""
    loop = NowcastLoop(warmed, interval=0.02)

    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.12)
    loop.stop()
    await task

    assert loop.stats.refreshes >= 3


# ---------------------------------------------------------------------- feed


def test_intra_bar_path_spans_the_bar(bars) -> None:
    bar = bars.iloc[0]
    path = _intra_bar_path(bar, 12)
    assert len(path) == 12
    assert path[0] == pytest.approx(float(bar["open"]))
    assert path[-1] == pytest.approx(float(bar["close"]))
    assert max(path) <= float(bar["high"]) + 1e-9
    assert min(path) >= float(bar["low"]) - 1e-9


def test_intra_bar_path_is_deterministic(bars) -> None:
    """A replay must be reproducible, or a bug found in one is not findable again."""
    bar = bars.iloc[3]
    assert _intra_bar_path(bar, 8) == _intra_bar_path(bar, 8)


def test_feed_paces_ticks_by_the_clock(bars) -> None:
    """One minute of wall clock per one-minute bar at speed 1."""
    feed = SimulatedFeed(bars.tail(20), on_tick=lambda tick: None, speed=1.0, ticks_per_bar=12)
    assert feed.tick_interval == pytest.approx(5.0)

    faster = SimulatedFeed(bars.tail(20), on_tick=lambda tick: None, speed=60.0, ticks_per_bar=12)
    assert faster.tick_interval == pytest.approx(5.0 / 60.0)


def test_feed_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SimulatedFeed(pd.DataFrame(), on_tick=lambda tick: None)


async def test_feed_streams_monotonic_ticks_from_now(bars) -> None:
    seen: list[Tick] = []
    feed = SimulatedFeed(
        bars.tail(40),
        on_tick=seen.append,
        speed=200_000.0,
        ticks_per_bar=4,
        bar_minutes=1,
    )
    started = datetime.now(IST)
    task = asyncio.create_task(feed.run())
    await asyncio.sleep(0.05)
    feed.stop()
    await task

    assert len(seen) > 8
    assert all(a.ts <= b.ts for a, b in zip(seen, seen[1:], strict=False))
    assert seen[0].ts >= started - timedelta(seconds=1)
    assert feed.status == "stopped"
    assert feed.ticks_emitted == len(seen)


async def test_feed_stops_promptly(bars) -> None:
    feed = SimulatedFeed(bars.tail(10), on_tick=lambda tick: None, speed=1.0)
    task = asyncio.create_task(feed.run())
    await asyncio.sleep(0.02)
    feed.stop()
    await asyncio.wait_for(task, timeout=2.0)


# -------------------------------------------------------------------- session


def test_session_composes_engine_and_renderer(offline_session) -> None:
    assert offline_session.engine is not None
    assert offline_session.renderer is not None
    assert offline_session.live is False
    assert "simulated" in offline_session.describe()
    assert "strategies" in offline_session.describe()


def test_session_bootstrap_warms_the_engine(offline_session) -> None:
    assert not offline_session.engine.history.empty
    assert offline_session.engine.projections, "bootstrap should leave a path ready to draw"
    assert offline_session.engine.forecaster.updates >= 1


def test_session_replay_produces_frames(offline_session, bars) -> None:
    """A short slice on purpose: each bar close rebuilds features and scores
    thirty strategies, so replaying three sessions would take minutes and prove
    nothing the first thirty bars do not.
    """
    frames = offline_session.replay(bars.tail(30))
    assert len(frames) > 5
    assert any(frame.projections for frame in frames)
    assert all(frame.candles is not None for frame in frames)


def test_session_reports_a_missing_renderer() -> None:
    kernel = Kernel(Settings())
    for entry in Kernel.bootstrap(Settings()).registry.find():
        if entry.kind is not PluginKind.RENDERER:
            kernel.register(entry)
    session = Session(kernel=kernel, config=SessionConfig(offline=True, days=2))
    assert session.renderer is None


def test_session_drives_a_simulated_feed(offline_session) -> None:
    """The feed the session picks offline is the clock-paced one, not a fast replay."""
    feed = offline_session._build_feed()
    assert isinstance(feed, SimulatedFeed)
    assert feed.speed == offline_session.config.speed


class RecordingRenderer:
    """A renderer that serves only inside its live block, like the web one.

    Which is the whole point: the socket is bound on entry and released on exit,
    so a frame pushed outside ``live()`` is a frame nobody can fetch.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.in_live_block = False

    @contextmanager
    def live(self):
        self.in_live_block = True
        self.events.append("live")
        try:
            yield
        finally:
            self.in_live_block = False
            self.events.append("closed")

    def live_update(self, snapshot, status: str = "") -> None:
        self.events.append("draw" if self.in_live_block else "draw-outside-live")


async def test_the_render_loop_draws_inside_the_live_block(offline_session) -> None:
    """The loop has to open the block, or the dashboard never appears at all.

    It pushed a frame a second into a renderer that was not serving, and left a
    process holding bootstrap output with nothing listening on the port — a
    running dashboard that answers nowhere, which is indistinguishable from a
    hang.
    """
    renderer = RecordingRenderer()
    session = Session(
        kernel=offline_session.kernel,
        config=SessionConfig(offline=True, days=2, refresh=0.05),
    )
    session.renderer = renderer

    task = asyncio.create_task(session._render_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert renderer.events[0] == "live", "the block must open before the first frame"
    assert "draw" in renderer.events
    assert "draw-outside-live" not in renderer.events
    assert renderer.events[-1] == "closed", "the port must be released"


def test_session_uses_the_broker_when_it_can() -> None:
    """With no credentials the session must fall back rather than raise."""

    class ConfiguredBroker:
        is_configured = True
        token = "fake"

        def feed(self, on_tick, on_status=None, instrument_keys=None, mode=None):
            return "live-feed"

        def close(self) -> None:
            pass

    kernel = Kernel(Settings())
    for entry in Kernel.bootstrap(Settings()).registry.find():
        if entry.handle == "source:upstox":
            continue
        kernel.register(entry)
    kernel.register(
        PluginEntry(
            manifest=PluginManifest(name="upstox", kind=PluginKind.SOURCE, provides=("broker",)),
            build=lambda ctx, **params: ConfiguredBroker(),
            module="tests.stub_broker",
        )
    )

    session = Session(kernel=kernel, config=SessionConfig(offline=False, days=2))
    assert session.live is True
    assert session._build_feed() == "live-feed"
