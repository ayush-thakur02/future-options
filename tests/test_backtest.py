"""Tests for validation splitting, the backtest engine, and cost accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from niftypulse.backtest import BacktestConfig, CostModel, run_backtest
from niftypulse.backtest.costs import net_expectancy
from niftypulse.ml.splits import purged_walk_forward

# ------------------------------------------------------------------- splits


def test_folds_are_chronological() -> None:
    """Training must always precede testing."""
    folds = purged_walk_forward(5000, n_splits=4, horizon=5, embargo=20)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train.max() < fold.test.min()


def test_purge_creates_a_gap() -> None:
    """The gap between train and test must cover the label window plus embargo."""
    horizon, embargo = 10, 25
    folds = purged_walk_forward(6000, n_splits=4, horizon=horizon, embargo=embargo)
    for fold in folds:
        gap = fold.test.min() - fold.train.max()
        assert gap >= horizon + embargo, f"gap {gap} is smaller than purge requirement"


def test_folds_do_not_overlap_in_test() -> None:
    folds = purged_walk_forward(5000, n_splits=4, horizon=5, embargo=20)
    covered: set[int] = set()
    for fold in folds:
        indices = set(fold.test.tolist())
        assert not (indices & covered), "test blocks overlap"
        covered |= indices


def test_walk_forward_training_window_expands() -> None:
    folds = purged_walk_forward(10_000, n_splits=5, horizon=5, embargo=20)
    sizes = [fold.train_size for fold in folds]
    assert sizes == sorted(sizes), "training window should expand or stay level"


def test_insufficient_data_raises() -> None:
    with pytest.raises(ValueError):
        purged_walk_forward(100, n_splits=6, horizon=5, embargo=20, min_train=500)


# ----------------------------------------------------------------- backtest


def _flat_bars(n: int = 2000) -> pd.DataFrame:
    """A deterministic ramp with 1-minute NSE session timestamps."""
    index = pd.date_range("2026-01-05 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    close = 24_000.0 + np.arange(n) * 0.5
    frame = pd.DataFrame(index=index)
    frame["open"] = close
    frame["high"] = close + 1.0
    frame["low"] = close - 1.0
    frame["close"] = close
    frame["volume"] = 0.0
    frame["oi"] = 0.0
    return frame


def test_engine_enters_at_next_bar_open() -> None:
    """Execution must use the first price available after the signal."""
    bars = _flat_bars(200)
    scores = pd.Series(0.0, index=bars.index)
    scores.iloc[50] = 0.9

    config = BacktestConfig(horizon=3, entry_threshold=0.15, notional=75_000, lot_size=75)
    result = run_backtest(bars, scores, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    expected_entry = bars["open"].iloc[51]
    expected_exit = bars["close"].iloc[54]
    assert trade.entry_price == pytest.approx(expected_entry)
    assert trade.exit_price == pytest.approx(expected_exit)
    assert trade.direction.value == "UP"


def test_engine_respects_threshold() -> None:
    bars = _flat_bars(200)
    scores = pd.Series(0.05, index=bars.index)
    config = BacktestConfig(horizon=3, entry_threshold=0.15)
    assert len(run_backtest(bars, scores, config).trades) == 0


def test_engine_does_not_overlap_positions() -> None:
    """Consecutive signals inside an open position must not open a second trade."""
    bars = _flat_bars(200)
    scores = pd.Series(0.9, index=bars.index)
    config = BacktestConfig(horizon=5, entry_threshold=0.15)
    result = run_backtest(bars, scores, config)

    for earlier, later in zip(result.trades, result.trades[1:], strict=False):
        assert later.entry_ts > earlier.exit_ts


def test_engine_blocks_direction_when_disabled() -> None:
    bars = _flat_bars(200)
    scores = pd.Series(-0.9, index=bars.index)
    config = BacktestConfig(horizon=3, entry_threshold=0.15, allow_short=False)
    assert len(run_backtest(bars, scores, config).trades) == 0


def test_costs_reduce_pnl() -> None:
    bars = _flat_bars(200)
    scores = pd.Series(0.0, index=bars.index)
    scores.iloc[50] = 0.9

    expensive = BacktestConfig(
        horizon=3, entry_threshold=0.15, cost_model=CostModel(slippage_bps=50.0)
    )
    cheap = BacktestConfig(
        horizon=3, entry_threshold=0.15, cost_model=CostModel(slippage_bps=0.0)
    )

    heavy = run_backtest(bars, scores, expensive).trades[0]
    light = run_backtest(bars, scores, cheap).trades[0]

    assert heavy.costs > light.costs
    assert heavy.net_pnl < light.net_pnl


def test_trade_direction_signs_pnl() -> None:
    """A short must profit when price falls."""
    bars = _flat_bars(200)
    falling = bars.copy()
    for column in ("open", "high", "low", "close"):
        falling[column] = 24_000.0 - np.arange(len(bars)) * 0.5

    scores = pd.Series(0.0, index=bars.index)
    scores.iloc[50] = -0.9
    config = BacktestConfig(horizon=3, entry_threshold=0.15, cost_model=CostModel())
    trade = run_backtest(falling, scores, config).trades[0]

    assert trade.direction.value == "DOWN"
    assert trade.gross_pnl > 0


def test_no_trades_on_empty_signal() -> None:
    bars = _flat_bars(200)
    scores = pd.Series(0.0, index=bars.index)
    result = run_backtest(bars, scores, BacktestConfig(horizon=3))
    assert result.trades == []
    assert result.report.trades == 0


def test_cost_model_scales_with_notional() -> None:
    model = CostModel()
    small = model.round_trip_bps(75_000)
    large = model.round_trip_bps(7_500_000)
    # Flat per-order brokerage makes small notionals relatively more expensive.
    assert small > large
    assert large > 0


def test_breakeven_move_is_positive() -> None:
    model = CostModel()
    assert model.breakeven_move_bps(1_000_000) > 0


def test_net_expectancy_subtracts_costs() -> None:
    model = CostModel()
    cost = model.round_trip_bps(1_000_000)
    assert net_expectancy(10.0, 1_000_000, model) == pytest.approx(10.0 - cost)


def test_position_sizing_targets_notional_when_affordable() -> None:
    """With a small enough lot, size lands on the requested exposure.

    A regression guard: sizing once treated the target notional as a unit count,
    so exposure came out at price x notional — roughly Rs 24 billion instead of
    Rs 1 million — and costs were overstated by four orders of magnitude.
    """
    bars = _flat_bars(200)
    scores = pd.Series(0.0, index=bars.index)
    scores.iloc[50] = 0.9

    config = BacktestConfig(
        horizon=3, entry_threshold=0.15, notional=1_000_000, lot_size=5
    )
    trade = run_backtest(bars, scores, config).trades[0]

    exposure = trade.entry_price * trade.quantity
    assert 0.9e6 < exposure < 1.1e6, f"exposure {exposure:,.0f} is not near the target"
    assert trade.quantity % 5 == 0, "size must be a whole number of lots"


def test_position_sizing_floors_at_one_lot() -> None:
    """A target below one lot's value must still trade exactly one lot.

    This is not a rounding nicety, it is the binding constraint on this
    instrument. NIFTY near 24,000 with a 75-unit lot makes the smallest possible
    position about Rs 1.8 million of notional, so any smaller target is simply
    unreachable — and a platform that silently pretended otherwise would report
    costs and PnL for a position nobody can hold.
    """
    bars = _flat_bars(200)
    scores = pd.Series(0.0, index=bars.index)
    scores.iloc[50] = 0.9

    config = BacktestConfig(
        horizon=3, entry_threshold=0.15, notional=100_000, lot_size=75
    )
    trade = run_backtest(bars, scores, config).trades[0]

    assert trade.quantity == 75
    one_lot = trade.entry_price * 75
    assert trade.entry_price * trade.quantity == pytest.approx(one_lot)
    assert one_lot > 1_000_000, "one lot should exceed the undersized target"


def test_costs_are_plausible_magnitude() -> None:
    """Round-trip cost should be single-digit basis points, not millions."""
    bars = _flat_bars(4000)
    rng = np.random.default_rng(3)
    scores = pd.Series(rng.uniform(-1, 1, len(bars)), index=bars.index)

    config = BacktestConfig(
        horizon=5, entry_threshold=0.5, notional=1_000_000, lot_size=75
    )
    report = run_backtest(bars, scores, config).report

    assert report.trades > 0
    assert 0 < report.cost_drag_bps < 50, f"cost drag {report.cost_drag_bps} bps is implausible"
    assert 0 < report.avg_notional < 2e6


def test_probability_series_is_not_valid_conviction() -> None:
    """Probabilities must be mapped to signed conviction before thresholding.

    A regression guard. The engine thresholds on signed conviction in [-1, 1].
    Feeding it probabilities instead means ``abs(0.48)`` — a model with no
    opinion — clears every threshold below 0.48, so the entry filter silently
    stops filtering and every bar trades.
    """
    bars = _flat_bars(400)
    # A model with essentially no view: probabilities hovering around 0.48.
    probabilities = pd.Series(0.48, index=bars.index)
    config = BacktestConfig(horizon=3, entry_threshold=0.15)

    as_probability = run_backtest(bars, probabilities, config)
    assert len(as_probability.trades) > 0, "raw probabilities will clear the gate"

    conviction = 2.0 * probabilities - 1.0  # -> -0.04, no opinion
    as_conviction = run_backtest(bars, conviction, config)
    assert len(as_conviction.trades) == 0, "mapped conviction must not clear the gate"


def test_report_metrics_are_consistent() -> None:
    bars = _flat_bars(2000)
    rng = np.random.default_rng(7)
    scores = pd.Series(rng.uniform(-1, 1, len(bars)), index=bars.index)
    result = run_backtest(bars, scores, BacktestConfig(horizon=5, entry_threshold=0.5))

    report = result.report
    if report.trades:
        assert report.wins + report.losses == report.trades
        assert 0.0 <= report.win_rate <= 1.0
        assert report.total_costs > 0
        # Gross must exceed net once costs are charged.
        assert report.avg_gross_bps > report.avg_net_bps
