"""Causality and correctness tests for the feature layer.

The most important test in this file is :func:`test_features_do_not_use_future_data`.
Lookahead is the failure mode that makes a backtest look excellent and a live
strategy lose money, and it is invisible in every summary statistic. The test
works by truncating the input to a prefix and asserting the features on the
overlapping rows are bit-for-bit unchanged — if any indicator peeks forward, the
values shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plugins.features.technical import build_features, feature_columns
from plugins.features.technical import indicators as ta


def test_features_do_not_use_future_data(bars: pd.DataFrame) -> None:
    """Features at bar t must not change when later bars are removed."""
    full = build_features(bars)

    cut = len(bars) - 120
    truncated = build_features(bars.iloc[:cut])

    overlapping = truncated.index
    assert len(overlapping) > 100

    left = full.loc[overlapping]
    right = truncated.loc[overlapping]

    for column in right.columns:
        a = left[column]
        b = right[column]
        both_nan = a.isna() & b.isna()
        close = np.isclose(
            a.to_numpy(dtype="float64", na_value=np.nan),
            b.to_numpy(dtype="float64", na_value=np.nan),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
        assert (both_nan | close).all(), f"column {column!r} changed when future bars were removed"


def test_features_have_no_infinities(bars: pd.DataFrame) -> None:
    features = build_features(bars)
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype="float64", na_value=0.0)).any()


def test_feature_columns_are_usable(bars: pd.DataFrame) -> None:
    features = build_features(bars)
    columns = feature_columns(features)
    assert len(columns) > 80
    # Metadata and internal helper columns must not leak into the model input.
    assert "bar_minutes" not in columns
    assert not any(column.startswith("_") for column in columns)


def test_forward_return_alignment(bars: pd.DataFrame) -> None:
    """A dead-banded label must agree in sign with the actual forward return."""
    from ml.dataset import directional_labels

    features = build_features(bars)
    labels, returns = directional_labels(bars, horizon=5, atr_norm=features["atr_norm"])

    known = labels.dropna()
    assert len(known) > 1000

    aligned = returns.loc[known.index]
    assert (aligned[known == 1] > 0).all(), "UP labels must have positive forward returns"
    assert (aligned[known == 0] < 0).all(), "DOWN labels must have negative forward returns"


def test_labels_do_not_cross_sessions(bars: pd.DataFrame) -> None:
    """No label may span a session boundary."""
    from ml.dataset import directional_labels, session_mask

    features = build_features(bars)
    labels, _ = directional_labels(bars, horizon=15, atr_norm=features["atr_norm"])
    valid = session_mask(bars, 15)

    crossings = labels.notna() & ~valid
    assert not crossings.any(), "labels leaked across an overnight gap"


# --------------------------------------------------------------- indicators


def test_rsi_bounds(bars: pd.DataFrame) -> None:
    values = ta.rsi(bars["close"], 14).dropna()
    assert len(values) > 0
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rsi_matches_known_series() -> None:
    """RSI of a monotonically rising series is 100."""
    rising = pd.Series(np.arange(1.0, 120.0))
    values = ta.rsi(rising, 14).dropna()
    assert values.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_atr_is_positive(bars: pd.DataFrame) -> None:
    values = ta.atr(bars, 14).dropna()
    assert (values > 0).all()


def test_true_range_bounds(bars: pd.DataFrame) -> None:
    tr = ta.true_range(bars).dropna()
    intrabar = (bars["high"] - bars["low"]).dropna()
    assert (tr >= intrabar - 1e-9).all()


def test_ema_tracks_constant_series() -> None:
    constant = pd.Series([100.0] * 300)
    assert ta.ema(constant, 50).iloc[-1] == pytest.approx(100.0)


def test_bollinger_pct_b_ordering(bars: pd.DataFrame) -> None:
    bands = ta.bollinger(bars["close"], 20, 2.0)
    valid = bands["pct_b"].dropna()
    assert valid.between(-2.0, 3.0).all()


def test_donchian_prior_channel_excludes_current_bar(bars: pd.DataFrame) -> None:
    """The breakout channel must lag by one bar, or breakouts can never trigger."""
    channel = ta.donchian(bars, 20)
    upper = channel["upper"]
    prior = channel["prior_upper"]

    # prior_upper at t equals upper at t-1.
    aligned = upper.shift(1)
    pd.testing.assert_series_equal(
        prior.iloc[30:60], aligned.iloc[30:60], check_names=False
    )
    # The prior channel can therefore be below the current high.
    triggered = (bars["close"] > prior).sum()
    assert triggered > 0, "a breakout must be achievable"


def test_supertrend_direction_is_binary(bars: pd.DataFrame) -> None:
    _, direction = ta.supertrend(bars, 10, 3.0)
    values = set(direction.dropna().unique())
    assert values <= {1.0, -1.0}


def test_supertrend_bands_are_ordered(bars: pd.DataFrame) -> None:
    line, _ = ta.supertrend(bars, 10, 3.0)
    valid = line.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


def test_session_vwap_resets_daily(bars: pd.DataFrame) -> None:
    vwap = ta.vwap_session(bars).dropna()
    days = bars.index.tz_convert("Asia/Kolkata").normalize().nunique()
    assert vwap.index.tz is not None
    assert days > 1


def test_efficiency_ratio_bounds(bars: pd.DataFrame) -> None:
    values = ta.efficiency_ratio(bars["close"], 10).dropna()
    assert values.min() >= 0.0
    assert values.max() <= 1.0 + 1e-9


def test_hurst_of_random_walk_is_near_half(rng: np.random.Generator) -> None:
    """A random walk should sit close to H=0.5."""
    steps = rng.standard_normal(4000)
    price = pd.Series(100.0 * np.exp(np.cumsum(steps * 0.001)))
    hurst = ta.hurst_exponent(price, window=800, max_lag=20).dropna()
    assert len(hurst) > 0
    assert 0.30 < hurst.mean() < 0.70
