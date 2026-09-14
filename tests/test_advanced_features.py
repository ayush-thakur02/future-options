from __future__ import annotations

import numpy as np

from plugins.features.technical import advanced as qa
from plugins.features.technical import build_features
from plugins.sources.simulated.series import generate_candles

ADVANCED_COLUMNS = {
    "kama_dist_10",
    "trix_15",
    "ppo_hist",
    "fisher_10",
    "rvi_10",
    "choppiness_14",
    "ulcer_14",
    "sortino_30",
    "autocorr_1_50",
    "variance_ratio_5_60",
    "direction_entropy_50",
    "ad_slope_20",
    "ease_of_movement_14",
    "amihud_20",
}


def test_advanced_indicators_are_exposed_as_model_features() -> None:
    bars = generate_candles(days=4, seed=71)
    features = build_features(bars)
    assert ADVANCED_COLUMNS <= set(features.columns)
    assert np.isfinite(features[list(ADVANCED_COLUMNS)].tail(50).to_numpy()).mean() > 0.95


def test_advanced_feature_values_do_not_change_when_future_is_removed() -> None:
    bars = generate_candles(days=5, seed=73)
    cutoff = 900
    full = build_features(bars)
    prefix = build_features(bars.iloc[:cutoff])
    for column in ADVANCED_COLUMNS:
        left = full[column].iloc[:cutoff].to_numpy()
        right = prefix[column].to_numpy()
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_kama_adapts_without_overshooting_a_monotonic_price_path() -> None:
    bars = generate_candles(days=2, seed=79)
    close = bars["close"].sort_index().cummax()
    average = qa.kama(close, window=10).dropna()
    aligned = close.reindex(average.index)
    assert (average <= aligned + 1e-9).all()
    assert average.index.equals(aligned.index)
