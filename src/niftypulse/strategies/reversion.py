"""Mean-reversion strategies.

Index intraday data mean-reverts more often than it trends, but only in the
absence of a strong trend. Each strategy here therefore scales its signal *down*
as trend strength rises — the naive version (buy every RSI below 30) gets run
over by exactly the days it fires most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyContext, squash


def _trend_dampener(features: pd.DataFrame, ceiling: float = 45.0, floor: float = 0.25) -> pd.Series:
    """Scale a reversion signal down as trend strength rises.

    The floor matters. RSI extremes and high ADX occur together almost by
    construction — a sustained decline is what produces both — so a dampener that
    can reach zero silences the strategy on exactly the bars it exists to trade.
    Shrinking the position instead of vetoing it keeps the strategy honest about
    the risk without switching it off.
    """
    if "adx_14" not in features.columns:
        return pd.Series(1.0, index=features.index, dtype="float64")
    strength = (features["adx_14"] / ceiling).clip(0.0, 1.0)
    return floor + (1.0 - floor) * (1.0 - strength)


class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"
    category = "reversion"
    description = "Fade RSI extremes, damped by trend strength"

    def __init__(self, low: float = 30.0, high: float = 70.0) -> None:
        self.low = low
        self.high = high

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "rsi_14" not in features.columns:
            return self._empty(context)

        rsi = features["rsi_14"]
        # Map (0, low) -> positive conviction, (high, 100) -> negative.
        oversold = ((self.low - rsi) / self.low).clip(lower=0.0)
        overbought = ((rsi - self.high) / (100.0 - self.high)).clip(lower=0.0)
        raw = (oversold - overbought) * 2.0

        return squash(raw * _trend_dampener(features), scale=0.9).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        rsi = context.features["rsi_14"].iloc[-1]
        return f"RSI {rsi:.0f} {'oversold bounce' if value > 0 else 'overbought fade'}"


class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"
    category = "reversion"
    description = "Fade closes outside the Bollinger band, damped by trend strength"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "bb_pct_b" not in features.columns:
            return self._empty(context)

        pct_b = features["bb_pct_b"]
        stretched_low = (0.05 - pct_b).clip(lower=0.0)
        stretched_high = (pct_b - 0.95).clip(lower=0.0)
        raw = (stretched_low - stretched_high) * 12.0

        return squash(raw * _trend_dampener(features), scale=0.8).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        pct_b = context.features["bb_pct_b"].iloc[-1]
        return f"Bollinger %B {pct_b:.2f}"


class VwapReversionStrategy(Strategy):
    name = "vwap_reversion"
    category = "reversion"
    description = "Fade extension away from session VWAP"

    def __init__(self, threshold: float = 0.0015) -> None:
        self.threshold = threshold

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "vwap_dist" not in features.columns:
            return self._empty(context)

        distance = features["vwap_dist"]
        # Only the part of the deviation beyond the threshold counts.
        excess = np.sign(distance) * (distance.abs() - self.threshold).clip(lower=0.0)
        raw = -excess / self.threshold

        return squash(raw * _trend_dampener(features), scale=0.9).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        distance = context.features["vwap_dist"].iloc[-1]
        return f"{abs(distance) * 10_000:.1f} bps from VWAP"


class ZScoreReversionStrategy(Strategy):
    name = "zscore_reversion"
    category = "reversion"
    description = "Fade price deviations from its 100-bar mean"

    def __init__(self, entry_z: float = 2.0) -> None:
        self.entry_z = entry_z

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "price_z_100" not in features.columns:
            return self._empty(context)

        z = features["price_z_100"]
        excess = np.sign(z) * (z.abs() - self.entry_z).clip(lower=0.0)
        raw = -excess / 1.5

        return squash(raw * _trend_dampener(features), scale=0.8).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        z = context.features["price_z_100"].iloc[-1]
        return f"Price at {z:+.1f} sigma from 100-bar mean"


class RangeFadeStrategy(Strategy):
    name = "range_fade"
    category = "reversion"
    description = "Fade extremes of the day's running range in low-volatility chop"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "day_range_position" not in features.columns:
            return self._empty(context)

        position = features["day_range_position"]
        at_high = ((position - 0.9) / 0.1).clip(0.0, 1.0)
        at_low = ((0.1 - position) / 0.1).clip(0.0, 1.0)
        raw = (at_low - at_high) * 1.5

        quiet = _trend_dampener(features, ceiling=25.0)
        return squash(raw * quiet, scale=0.8).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        position = context.features["day_range_position"].iloc[-1]
        return f"At {position:.0%} of the day's range"
