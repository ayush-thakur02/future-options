"""Tests for data ingestion: aggregation, resampling, and the candle store.

The store's own partitioning behaviour has a test module of its own; what is here
is the part that belongs to ingestion — that what comes out matches what went in.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.bars import normalize_candles
from core.calendar import IST, TradingCalendar
from core.types import Direction, Tick
from plugins.aggregators.candle_builder import CandleAggregator, bar_start
from plugins.sources.history import PartitionedStore
from plugins.sources.history.resample import resample_ohlcv
from plugins.sources.simulated.series import generate_candles


def make_tick(minute: int, price: float, second: int = 0) -> Tick:
    ts = datetime(2026, 1, 5, 9, 15, tzinfo=IST) + timedelta(minutes=minute, seconds=second)
    return Tick(ts=ts, ltp=price, ltq=10, volume_traded=1000, bid_p=price - 0.05, ask_p=price + 0.05)


# --------------------------------------------------------------- bar bucketing


def test_bar_start_anchors_to_session_open() -> None:
    """Bars align to 09:15, not to the top of the hour."""
    moment = datetime(2026, 1, 5, 9, 17, 30, tzinfo=IST)
    start = bar_start(moment, 5)
    assert start.hour == 9
    assert start.minute == 15

    later = datetime(2026, 1, 5, 9, 22, 0, tzinfo=IST)
    assert bar_start(later, 5).minute == 20


def test_bar_start_handles_one_minute_bars() -> None:
    moment = datetime(2026, 1, 5, 14, 37, 45, tzinfo=IST)
    start = bar_start(moment, 1)
    assert (start.hour, start.minute, start.second) == (14, 37, 0)


# ------------------------------------------------------------------ aggregator


def test_aggregator_builds_ohlc() -> None:
    aggregator = CandleAggregator(bar_minutes=1)
    prices = [100.0, 102.0, 99.0, 101.0]

    closed = None
    for i, price in enumerate(prices):
        closed = aggregator.on_tick(make_tick(i // 4, price, second=i * 10))

    bar = aggregator.current_bar
    assert bar is not None
    assert bar["open"] == pytest.approx(100.0)
    assert bar["high"] == pytest.approx(102.0)
    assert bar["low"] == pytest.approx(99.0)
    assert bar["close"] == pytest.approx(101.0)
    assert closed is None  # still inside the first bar


def test_aggregator_emits_on_rollover() -> None:
    aggregator = CandleAggregator(bar_minutes=1)
    aggregator.on_tick(make_tick(0, 100.0))
    aggregator.on_tick(make_tick(0, 101.0, second=30))

    closed = aggregator.on_tick(make_tick(1, 102.0))

    assert closed is not None
    assert closed["open"] == pytest.approx(100.0)
    assert closed["close"] == pytest.approx(101.0)
    assert aggregator.current_bar["open"] == pytest.approx(102.0)


def test_aggregator_counts_ticks() -> None:
    """Tick count is the activity proxy when an index has no volume."""
    aggregator = CandleAggregator(bar_minutes=1)
    for i in range(7):
        aggregator.on_tick(make_tick(0, 100.0 + i, second=i))

    assert aggregator.current_bar["tick_count"] == 7


def test_aggregator_does_not_emit_across_sessions() -> None:
    """A partial bar must be discarded rather than spanning an overnight gap."""
    aggregator = CandleAggregator(bar_minutes=1)
    aggregator.on_tick(make_tick(0, 100.0))

    next_day = datetime(2026, 1, 6, 9, 16, tzinfo=IST)
    closed = aggregator.on_tick(Tick(ts=next_day, ltp=105.0))

    assert closed is None
    assert aggregator.current_bar["open"] == pytest.approx(105.0)


def test_aggregator_snapshot_frame_shape(bars: pd.DataFrame) -> None:
    aggregator = CandleAggregator(bar_minutes=1)
    for i in range(5):
        aggregator.on_tick(make_tick(i, 100.0 + i))

    frame = aggregator.snapshot_frame()
    assert "tick_count" in frame.columns
    assert frame.index.tz is not None


# ------------------------------------------------------------------ resampling


def test_resample_preserves_price_extremes(bars: pd.DataFrame) -> None:
    resampled = resample_ohlcv(bars, 5)
    assert not resampled.empty
    assert resampled["high"].max() <= bars["high"].max() + 1e-9
    assert resampled["low"].min() >= bars["low"].min() - 1e-9
    assert resampled["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])


def test_resample_five_minute_has_fewer_bars(bars: pd.DataFrame) -> None:
    resampled = resample_ohlcv(bars, 5)
    assert len(resampled) < len(bars)
    assert len(resampled) > len(bars) / 6


def test_resample_does_not_merge_sessions(bars: pd.DataFrame) -> None:
    """No aggregated bar may contain data from two different trading days."""
    resampled = resample_ohlcv(bars, 15)
    days = resampled.index.normalize().nunique()
    assert days == bars.index.normalize().nunique()


def test_resample_is_identity_for_one_minute(bars: pd.DataFrame) -> None:
    resampled = resample_ohlcv(bars, 1)
    assert len(resampled) == len(bars)


# ----------------------------------------------------------------------- store


def test_store_roundtrip(tmp_path, bars: pd.DataFrame) -> None:
    store = PartitionedStore(tmp_path, "NSE_INDEX|Nifty 50")
    store.write(bars)

    loaded = store.load()
    assert len(loaded) == len(bars)
    assert loaded["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])


def test_store_deduplicates(tmp_path, bars: pd.DataFrame) -> None:
    store = PartitionedStore(tmp_path, "NSE_INDEX|Nifty 50")
    store.write(bars)
    store.write(bars)

    assert store.row_count() == len(bars)


def test_store_append_updates_existing(tmp_path) -> None:
    """Re-fetching an overlapping window must overwrite, not duplicate."""
    index = pd.date_range("2026-01-05 09:15", periods=10, freq="1min", tz=IST)
    first = normalize_candles(
        pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0, "oi": 0.0},
            index=index,
        )
    )
    store = PartitionedStore(tmp_path, "NSE_INDEX|Nifty 50")
    store.write(first)

    revised = first.copy()
    revised["close"] = 2.0
    store.write(revised)

    assert store.row_count() == 10
    assert store.load()["close"].iloc[0] == pytest.approx(2.0)


def test_store_slice(tmp_path, bars: pd.DataFrame) -> None:
    store = PartitionedStore(tmp_path, "NSE_INDEX|Nifty 50")
    store.write(bars)

    start = bars.index[100]
    end = bars.index[200]
    sliced = store.load(start=start, end=end)
    assert len(sliced) == 101
    assert sliced.index[0] == start


def test_normalize_adds_missing_columns() -> None:
    index = pd.date_range("2026-01-05 09:15", periods=3, freq="1min", tz=IST)
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
    normalized = normalize_candles(frame)
    assert "open" in normalized.columns
    assert "volume" in normalized.columns
    assert len(normalized) == 3


# -------------------------------------------------------------------- models


def test_tick_derived_measures() -> None:
    tick = Tick(ts=datetime.now(IST), ltp=100.0, bid_p=99.5, bid_q=300, ask_p=100.5, ask_q=100)
    assert tick.mid == pytest.approx(100.0)
    assert tick.spread == pytest.approx(1.0)
    assert tick.depth_imbalance > 0  # bid-heavy

    balanced = Tick(ts=datetime.now(IST), ltp=100.0, bid_q=100, ask_q=100)
    assert balanced.depth_imbalance == pytest.approx(0.0)


def test_direction_from_value() -> None:
    assert Direction.from_value(0.01, deadband=0.005) is Direction.UP
    assert Direction.from_value(-0.01, deadband=0.005) is Direction.DOWN
    assert Direction.from_value(0.001, deadband=0.005) is Direction.FLAT


# ------------------------------------------------------------------ calendar


def test_calendar_weekend_is_closed() -> None:
    calendar = TradingCalendar()
    saturday = datetime(2026, 1, 10, 11, 0, tzinfo=IST)
    assert not calendar.is_open(saturday)


def test_calendar_session_hours() -> None:
    calendar = TradingCalendar()
    assert calendar.is_open(datetime(2026, 1, 5, 10, 0, tzinfo=IST))
    assert not calendar.is_open(datetime(2026, 1, 5, 9, 0, tzinfo=IST))
    assert not calendar.is_open(datetime(2026, 1, 5, 16, 0, tzinfo=IST))


def test_calendar_respects_holidays() -> None:
    from datetime import date

    calendar = TradingCalendar(holidays={date(2026, 1, 26)})
    assert not calendar.is_open(datetime(2026, 1, 26, 11, 0, tzinfo=IST))


def test_session_progress_bounds() -> None:
    calendar = TradingCalendar()
    assert calendar.session_progress(datetime(2026, 1, 5, 9, 15, tzinfo=IST)) == pytest.approx(0.0)
    assert calendar.session_progress(datetime(2026, 1, 5, 15, 30, tzinfo=IST)) == pytest.approx(1.0)
    assert 0.0 < calendar.session_progress(datetime(2026, 1, 5, 12, 0, tzinfo=IST)) < 1.0


def test_next_open_skips_weekend() -> None:
    calendar = TradingCalendar()
    friday_close = datetime(2026, 1, 9, 16, 0, tzinfo=IST)
    nxt = calendar.next_open(friday_close)
    assert nxt.weekday() == 0  # Monday


# ----------------------------------------------------------------- synthetic


def test_synthetic_bars_are_well_formed() -> None:
    frame = generate_candles(days=5, seed=1)
    assert (frame["high"] >= frame["low"]).all()
    assert (frame["high"] >= frame["close"] - 1e-6).all()
    assert (frame["low"] <= frame["close"] + 1e-6).all()
    assert frame.index.tz is not None


def test_synthetic_is_deterministic() -> None:
    a = generate_candles(days=3, seed=42)
    b = generate_candles(days=3, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_different_seeds_differ() -> None:
    a = generate_candles(days=3, seed=1)
    b = generate_candles(days=3, seed=2)
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
