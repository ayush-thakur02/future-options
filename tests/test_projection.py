"""Tests for the projected candles.

This is the feature the dashboard leans on hardest, so the tests cover the
properties a reader would notice if they broke: the path starts where the market
actually is, it leans the way conviction points, it decays and widens with
horizon, and the scoreboard that judges it counts hits and misses correctly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.calendar import IST
from core.types import Tick
from plugins.forecasts.projection import (
    MicroMomentum,
    ProjectionForecaster,
    ProjectionTracker,
    atr_of,
    project_candles,
    projected_frame,
    volatility_scale,
)
from plugins.sources.simulated.series import generate_candles

ANCHOR = 24_000.0
NOW = datetime(2026, 3, 10, 11, 0, tzinfo=IST)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return generate_candles(days=3, seed=21)


def tick(price: float, moment: datetime) -> Tick:
    return Tick(ts=moment, ltp=price)


# ------------------------------------------------------------------- geometry


def test_projection_returns_the_requested_number_of_bars(bars) -> None:
    assert len(project_candles(bars, ANCHOR, 0.5, now=NOW)) == 3
    assert len(project_candles(bars, ANCHOR, 0.5, now=NOW, bars_ahead=5)) == 5


def test_projection_is_empty_when_asked_for_nothing(bars) -> None:
    assert project_candles(bars, ANCHOR, 0.5, now=NOW, bars_ahead=0) == []


def test_projection_starts_at_the_live_price(bars) -> None:
    """The first projected bar opens where the tape is, not where the last bar closed.

    This is what makes the projected candles track the market between bar closes
    instead of drifting away from it.
    """
    moved = ANCHOR + 37.0
    projections = project_candles(bars, moved, 0.4, now=NOW)
    assert projections[0].open == pytest.approx(moved)
    assert projections[0].open != pytest.approx(float(bars["close"].iloc[-1]))


def test_projection_bars_are_consecutive_and_start_after_now(bars) -> None:
    projections = project_candles(bars, ANCHOR, 0.4, now=NOW, bar_minutes=1)
    stamps = [candle.ts for candle in projections]
    assert stamps[0] == NOW.replace(minute=1)  # 11:00 -> the 11:01 bar
    assert stamps == [stamps[0] + timedelta(minutes=i) for i in range(len(stamps))]


def test_projection_chains_open_to_the_previous_close(bars) -> None:
    """Bar two opens where bar one closed, as a real series does."""
    projections = project_candles(bars, ANCHOR, 0.7, now=NOW, bars_ahead=3)
    for previous, following in zip(projections, projections[1:], strict=False):
        assert following.open == pytest.approx(previous.close)


def test_every_projected_candle_contains_its_own_body(bars) -> None:
    for candle in project_candles(bars, ANCHOR, -0.8, now=NOW, bars_ahead=4):
        assert candle.high >= max(candle.open, candle.close)
        assert candle.low <= min(candle.open, candle.close)
        assert candle.low > 0


# ------------------------------------------------------------------ direction


def test_positive_conviction_projects_a_rising_path(bars) -> None:
    projections = project_candles(bars, ANCHOR, 0.8, now=NOW)
    closes = [candle.close for candle in projections]
    assert closes == sorted(closes)
    assert closes[-1] > ANCHOR


def test_negative_conviction_projects_a_falling_path(bars) -> None:
    projections = project_candles(bars, ANCHOR, -0.8, now=NOW)
    closes = [candle.close for candle in projections]
    assert closes == sorted(closes, reverse=True)
    assert closes[-1] < ANCHOR


def test_no_conviction_projects_a_flat_path(bars) -> None:
    """With no view the path must not invent one."""
    projections = project_candles(bars, ANCHOR, 0.0, now=NOW)
    for candle in projections:
        assert candle.close == pytest.approx(ANCHOR)
        assert candle.open == pytest.approx(ANCHOR)


def test_conviction_magnitude_scales_the_drift(bars) -> None:
    """Half the conviction, half the move — the mapping has to be monotone."""
    weak = project_candles(bars, ANCHOR, 0.4, now=NOW)[0].close - ANCHOR
    strong = project_candles(bars, ANCHOR, 0.8, now=NOW)[0].close - ANCHOR
    assert strong == pytest.approx(weak * 2, rel=1e-6)


def test_conviction_is_clamped_to_a_full_view(bars) -> None:
    """A conviction above 1 must not project beyond a full-conviction path."""
    full = project_candles(bars, ANCHOR, 1.0, now=NOW)[-1].close
    beyond = project_candles(bars, ANCHOR, 5.0, now=NOW)[-1].close
    assert beyond == pytest.approx(full)


def test_conviction_decays_with_horizon(bars) -> None:
    """The third bar must claim less than the first, not the same."""
    projections = project_candles(bars, ANCHOR, 1.0, now=NOW, bars_ahead=3)
    moves = [abs(candle.close - candle.open) for candle in projections]
    assert moves[0] > moves[1] > moves[2]
    assert projections[0].confidence > projections[1].confidence > projections[2].confidence


def test_projected_range_widens_with_horizon(bars) -> None:
    """Uncertainty compounds, so the far bars are visibly less certain."""
    projections = project_candles(bars, ANCHOR, 0.0, now=NOW, bars_ahead=3)
    spans = [candle.high - candle.low for candle in projections]
    assert spans == sorted(spans)
    assert spans[-1] > spans[0]


def test_expected_move_is_reported_in_basis_points(bars) -> None:
    candle = project_candles(bars, ANCHOR, 0.5, now=NOW)[0]
    expected = (candle.close / candle.open - 1.0) * 10_000
    assert candle.expected_move_bps == pytest.approx(expected)


# ----------------------------------------------------------------- volatility


def test_atr_is_positive_and_scales_with_range() -> None:
    calm = pd.DataFrame(
        {"open": [100.0] * 40, "high": [100.5] * 40, "low": [99.5] * 40, "close": [100.0] * 40}
    )
    wild = pd.DataFrame(
        {"open": [100.0] * 40, "high": [110.0] * 40, "low": [90.0] * 40, "close": [100.0] * 40}
    )
    assert atr_of(calm) == pytest.approx(1.0, abs=0.01)
    assert atr_of(wild) > atr_of(calm)


def test_atr_survives_an_empty_or_flat_series() -> None:
    """A flat series has no range; the projection must still be finite."""
    flat = pd.DataFrame(
        {"open": [24_000.0] * 30, "high": [24_000.0] * 30, "low": [24_000.0] * 30, "close": [24_000.0] * 30}
    )
    assert atr_of(flat) > 0
    projections = project_candles(flat, 24_000.0, 0.5, now=NOW)
    assert all(candle.high > candle.low for candle in projections)


def test_atr_of_an_empty_frame_is_zero() -> None:
    assert atr_of(pd.DataFrame(columns=["open", "high", "low", "close"])) == 0.0


def test_volatility_scale_is_clamped(bars) -> None:
    """One violent bar must not project an absurd range."""
    quiet = bars.tail(60).copy()
    scale = volatility_scale(quiet, atr=10_000.0)
    assert 0.55 <= scale <= 2.20


def test_volatility_scale_falls_back_when_atr_is_zero(bars) -> None:
    assert volatility_scale(bars, atr=0.0) == 1.0


def test_wider_volatility_projects_a_wider_range(bars) -> None:
    narrow = project_candles(bars, ANCHOR, 0.0, now=NOW)[0]
    wide = project_candles(bars * 2, ANCHOR, 0.0, now=NOW)[0]
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


# -------------------------------------------------------------------- frames


def test_projected_frame_matches_the_projection(bars) -> None:
    projections = project_candles(bars, ANCHOR, 0.3, now=NOW, bars_ahead=3)
    frame = projected_frame(projections)
    assert len(frame) == 3
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "oi"]
    assert frame["close"].iloc[-1] == pytest.approx(projections[-1].close)
    assert str(frame.index.tz) == str(IST)


def test_projected_frame_of_nothing_is_empty(bars) -> None:
    assert projected_frame([]).empty


def test_path_moves_when_the_anchor_moves(bars) -> None:
    """The fluctuation the dashboard shows: a new tick re-bases the whole path."""
    first = project_candles(bars, ANCHOR, 0.5, now=NOW)[0].close
    second = project_candles(bars, ANCHOR + 12.0, 0.5, now=NOW)[0].close
    assert second - first == pytest.approx(12.0)


# ------------------------------------------------------------------ momentum


def test_momentum_is_zero_until_prices_diverge() -> None:
    momentum = MicroMomentum()
    assert momentum.value(atr=10.0) == 0.0
    momentum.update(tick(24_000.0, NOW))
    assert momentum.value(atr=10.0) == 0.0


def test_momentum_turns_positive_on_a_rising_tape() -> None:
    momentum = MicroMomentum()
    for step in range(40):
        momentum.update(tick(24_000.0 + step * 0.5, NOW + timedelta(seconds=step)))
    assert momentum.value(atr=5.0) > 0.3
    assert momentum.spread_bps > 0


def test_momentum_turns_negative_on_a_falling_tape() -> None:
    momentum = MicroMomentum()
    for step in range(40):
        momentum.update(tick(24_000.0 - step * 0.5, NOW + timedelta(seconds=step)))
    assert momentum.value(atr=5.0) < -0.3
    assert momentum.spread_bps < 0


def test_momentum_is_bounded() -> None:
    """Saturation is what stops one large print swinging the whole path."""
    momentum = MicroMomentum()
    for step in range(30):
        momentum.update(tick(24_000.0 + step * 500.0, NOW + timedelta(seconds=step)))
    assert -1.0 <= momentum.value(atr=1.0) <= 1.0


def test_momentum_tightens_as_volatility_rises() -> None:
    """The same tape is a weaker signal in a fast market than in a quiet one."""
    momentum = MicroMomentum()
    for step in range(30):
        momentum.update(tick(24_000.0 + step * 0.4, NOW + timedelta(seconds=step)))
    assert momentum.value(atr=2.0) > momentum.value(atr=20.0)


def test_momentum_ignores_zero_prices() -> None:
    momentum = MicroMomentum()
    momentum.update(tick(0.0, NOW))
    assert momentum.ticks == 0
    assert momentum.value(atr=10.0) == 0.0


def test_momentum_responds_to_sustained_drift_not_a_burst() -> None:
    """A burst of prints at one instant must not read as a lasting trend.

    Twenty prints spanning 100ms and twenty spanning 20 seconds cover the same
    distance; only the second is a move the market actually travelled. Weighting
    by elapsed time is what separates them.
    """
    burst = MicroMomentum()
    burst.update(tick(24_000.0, NOW))
    for step in range(20):
        burst.update(tick(24_010.0, NOW + timedelta(milliseconds=step * 5)))

    gradual = MicroMomentum()
    for step in range(21):
        gradual.update(tick(24_000.0 + step * 0.5, NOW + timedelta(seconds=step)))

    assert burst.spread_bps < gradual.spread_bps
    assert abs(burst.value(atr=5.0)) < abs(gradual.value(atr=5.0))


def test_momentum_reset_clears_state() -> None:
    momentum = MicroMomentum()
    momentum.update(tick(24_000.0, NOW))
    momentum.update(tick(24_010.0, NOW + timedelta(seconds=1)))
    momentum.reset()
    assert momentum.ticks == 0
    assert momentum.value(atr=5.0) == 0.0


# ------------------------------------------------------------------- tracker


def test_tracker_scores_a_hit_when_the_bar_closes_the_projected_way(bars) -> None:
    tracker = ProjectionTracker()
    projections = project_candles(bars, ANCHOR, 0.9, now=NOW)
    tracker.record(projections, ANCHOR)

    first_target = pd.Timestamp(projections[0].ts)
    scores = tracker.observe(first_target, ANCHOR + 8.0)

    assert len(scores) == 1
    assert scores[0].hit is True
    assert scores[0].horizon == 1
    assert scores[0].error_bps > 0


def test_tracker_scores_a_miss_when_the_bar_goes_the_other_way(bars) -> None:
    tracker = ProjectionTracker()
    tracker.record(project_candles(bars, ANCHOR, 0.9, now=NOW), ANCHOR)
    scores = tracker.observe(pd.Timestamp(project_candles(bars, ANCHOR, 0.9, now=NOW)[0].ts), ANCHOR - 8.0)
    assert scores[0].hit is False


def test_tracker_keeps_per_horizon_statistics(bars) -> None:
    """Each horizon is scored when its own bar closes, so the three are distinct."""
    tracker = ProjectionTracker()
    projections = project_candles(bars, ANCHOR, 0.9, now=NOW, bars_ahead=3)
    tracker.record(projections, ANCHOR)

    for candle in projections:
        tracker.observe(pd.Timestamp(candle.ts), ANCHOR + 5.0)

    stats = tracker.stats()
    assert sorted(stats) == [1, 2, 3]
    assert all(stat.scored == 1 and stat.hits == 1 for stat in stats.values())
    assert tracker.overall().scored == 3
    assert "directional hit" in tracker.summary()


def test_tracker_scores_the_latest_record_for_a_target(bars) -> None:
    """A fresher projection for the same bar and horizon replaces the stale one."""
    tracker = ProjectionTracker()
    target = pd.Timestamp(project_candles(bars, ANCHOR, 1.0, now=NOW)[0].ts)

    tracker.record(project_candles(bars, ANCHOR, 1.0, now=NOW, bars_ahead=1), ANCHOR)
    tracker.record(project_candles(bars, ANCHOR, -1.0, now=NOW, bars_ahead=1), ANCHOR)
    scores = tracker.observe(target, ANCHOR - 5.0)

    assert len(scores) == 1
    assert scores[0].hit is True, "the down-turned projection should have been the one kept"


def test_tracker_prunes_targets_that_never_closed(bars) -> None:
    """A gap in the feed must not leave stale projections to be scored later."""
    tracker = ProjectionTracker()
    tracker.record(project_candles(bars, ANCHOR, 0.5, now=NOW, bars_ahead=3), ANCHOR)
    later = pd.Timestamp(NOW) + timedelta(minutes=30)
    assert tracker.observe(later, ANCHOR + 1.0) == []
    assert tracker.pending == 0


def test_tracker_reports_nothing_before_anything_closes() -> None:
    tracker = ProjectionTracker()
    assert tracker.summary() == "no projections scored yet"
    assert tracker.overall().scored == 0


def test_tracker_ignores_anchorless_records(bars) -> None:
    tracker = ProjectionTracker()
    tracker.record(project_candles(bars, ANCHOR, 0.5, now=NOW), anchor=0.0)
    assert tracker.pending == 0


# --------------------------------------------------------------- forecaster


def test_forecaster_refreshes_and_records(bars) -> None:
    forecaster = ProjectionForecaster(bars_ahead=3)
    projections = forecaster.refresh(bars, ANCHOR, conviction=0.6, now=NOW)

    assert len(projections) == 3
    assert forecaster.updates == 1
    assert forecaster.tracker.pending == 3
    assert forecaster.atr > 0
    assert forecaster.anchor == ANCHOR


def test_forecaster_blends_live_tick_momentum_into_the_path(bars) -> None:
    """Without this the projected candles would sit still between bar closes."""
    quiet = ProjectionForecaster(bars_ahead=1)
    moving = ProjectionForecaster(bars_ahead=1)

    for step in range(40):
        moving.momentum.update(tick(24_000.0 + step * 0.6, NOW + timedelta(seconds=step)))

    flat_path = quiet.refresh(bars, ANCHOR, conviction=0.0, now=NOW)
    moving_path = moving.refresh(bars, ANCHOR, conviction=0.0, now=NOW)

    assert quiet.micro == pytest.approx(0.0)
    assert moving.micro > 0.0
    assert moving_path[0].close > flat_path[0].close


def test_forecaster_momentum_weight_is_bounded(bars) -> None:
    forecaster = ProjectionForecaster(micro_weight=5.0)
    assert forecaster.micro_weight == 1.0


def test_forecaster_scores_the_bar_that_closed(bars) -> None:
    forecaster = ProjectionForecaster(bars_ahead=1)
    projections = forecaster.refresh(bars, ANCHOR, conviction=0.5, now=NOW)
    forecaster.observe_bar(pd.Timestamp(projections[0].ts), ANCHOR + 4.0)
    assert forecaster.tracker.overall().scored == 1


def test_forecaster_describes_itself(bars) -> None:
    forecaster = ProjectionForecaster(bars_ahead=3)
    assert forecaster.summary == "no projection"
    forecaster.refresh(bars, ANCHOR, conviction=0.9, now=NOW)
    assert "3 bars" in forecaster.summary
    described = forecaster.describe()
    assert described["bars_ahead"] == 3
    assert "tracking" in described


def test_atr_matches_the_pandas_version(bars) -> None:
    """The array path and the frame path have to agree on every shape.

    `atr_of` was rewritten without pandas because it runs once per projected bar
    per instrument. The trap is the first bar, which has no previous close:
    `DataFrame.max(axis=1)` skips that NaN, while `numpy.maximum` would propagate
    it and quietly drop the bar from the mean.
    """
    import numpy as np

    def pandas_atr(frame: pd.DataFrame, window: int = 14) -> float:
        if frame.empty:
            return 0.0
        high = frame["high"].astype("float64")
        low = frame["low"].astype("float64")
        close = frame["close"].astype("float64")
        previous = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
        ).max(axis=1)
        value = float(true_range.tail(window).mean())
        if not np.isfinite(value) or value <= 0.0:
            price = float(close.iloc[-1]) if len(close) else 0.0
            return price * 1e-5
        return value

    for length in (1, 2, 3, 14, 15, 60, 200):
        window = bars.tail(length)
        assert atr_of(window) == pytest.approx(pandas_atr(window), rel=1e-12), (
            f"atr_of disagreed with pandas over {length} bars"
        )

    flat = bars.tail(30).copy()
    for column in ("open", "high", "low", "close"):
        flat[column] = 24_000.0
    assert atr_of(flat) == pandas_atr(flat) == 24_000.0 * 1e-5
