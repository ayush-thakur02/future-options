"""Channel strategies that measure where price is being accepted.

Donchian's one-bar new-high event already has a strategy in the trend pack. These
rules answer different questions: whether price can hold outside an ATR envelope,
whether it has escaped a fitted trend channel, and whether closes persistently
auction near one side of a rolling range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, require_columns, squash

EPS = 1e-12


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope for each trailing window without Python callbacks."""
    result = np.full(len(series), np.nan, dtype="float64")
    if len(series) < window:
        return pd.Series(result, index=series.index, dtype="float64")

    values = series.to_numpy(dtype="float64")
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    x = np.arange(window, dtype="float64")
    centered = x - x.mean()
    slopes = windows @ centered / float((centered**2).sum())
    slopes[~np.isfinite(windows).all(axis=1)] = np.nan
    result[window - 1 :] = slopes
    return pd.Series(result, index=series.index, dtype="float64")


class KeltnerContinuationStrategy(Strategy):
    name = "keltner_continuation"
    category = "channels"
    description = "Follow acceptance outside a Keltner envelope when trend quality confirms"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["keltner_position", "adx_14", "efficiency_ratio_10"]
        if not require_columns(features, needed):
            return self._empty(context)

        # Position is zero at the lower band, 0.5 at the EMA and one at the upper
        # band. Ignore ordinary movement inside the middle 70% of the envelope.
        displacement = (features["keltner_position"] - 0.5) * 2.0
        outside = np.sign(displacement) * (displacement.abs() - 0.70).clip(lower=0.0)
        adx_gate = ((features["adx_14"] - 18.0) / 17.0).clip(0.0, 1.0)
        efficient = ((features["efficiency_ratio_10"] - 0.18) / 0.47).clip(0.0, 1.0)
        raw = outside * (0.25 + 0.75 * np.minimum(adx_gate, efficient))
        return squash(raw, scale=0.35).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        position = float(context.features["keltner_position"].iloc[-1])
        return f"Keltner position {position:.2f}, {'upper' if value > 0 else 'lower'} acceptance"


class RegressionChannelBreakoutStrategy(Strategy):
    name = "regression_channel_breakout"
    category = "channels"
    description = "Break from a rolling least-squares channel, scaled by fit and slope agreement"

    def __init__(self, window: int = 30, entry_z: float = 1.35) -> None:
        self.window = window
        self.entry_z = entry_z

    def score(self, context: StrategyContext) -> pd.Series:
        bars = context.bars
        if bars.empty or "close" not in bars.columns:
            return self._empty(context)

        close = bars["close"].astype("float64").reindex(context.features.index)
        raw_slope = _rolling_slope(close, self.window)
        endpoint_offset = (self.window - 1) / 2.0
        mean = close.rolling(self.window, min_periods=self.window).mean()
        endpoint = mean + raw_slope * endpoint_offset
        residual = close - endpoint
        noise = residual.rolling(self.window, min_periods=self.window).std(ddof=0)
        channel_z = residual / (noise + EPS)

        excess = np.sign(channel_z) * (channel_z.abs() - self.entry_z).clip(lower=0.0)
        slope_agrees = (np.sign(raw_slope) == np.sign(channel_z)).astype("float64")
        quality = context.features.get("r2_20", pd.Series(0.5, index=close.index)).clip(0.0, 1.0)
        raw = excess * (0.35 + 0.35 * slope_agrees + 0.30 * quality)
        return squash(raw, scale=1.0).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Regression channel expansion {'above' if value > 0 else 'below'} fitted trend"


class ChannelPressureStrategy(Strategy):
    name = "channel_pressure"
    category = "channels"
    description = "Sustained closes near one side of a rolling range, gated by directional strength"

    def __init__(self, persistence: int = 8) -> None:
        self.persistence = persistence

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["donchian_position", "di_spread", "adx_14"]
        if not require_columns(features, needed):
            return self._empty(context)

        location = ((features["donchian_position"] - 0.5) * 2.0).clip(-1.0, 1.0)
        pressure = location.rolling(self.persistence, min_periods=self.persistence).mean()
        directional = squash(features["di_spread"], scale=18.0)
        agreement = (np.sign(pressure) == np.sign(directional)).astype("float64")
        trend_gate = ((features["adx_14"] - 15.0) / 20.0).clip(0.0, 1.0)
        raw = pressure * agreement * (0.35 + 0.65 * trend_gate)
        return squash(raw, scale=0.55).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Closes persist near the channel {'high' if value > 0 else 'low'}"


STRATEGIES = (
    KeltnerContinuationStrategy,
    RegressionChannelBreakoutStrategy,
    ChannelPressureStrategy,
)
