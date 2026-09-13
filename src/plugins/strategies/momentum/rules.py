"""Momentum and order-flow strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, squash


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"
    category = "momentum"
    description = "MACD histogram level and slope, normalised by price"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "macd_hist" not in features.columns:
            return self._empty(context)

        hist = features["macd_hist"]
        slope = hist.diff(3)
        # Level says where we are; slope says whether it is still building.
        raw = squash(hist, scale=0.0004) * 0.6 + squash(slope, scale=0.0003) * 0.4
        return squash(raw, scale=0.8).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        hist = context.features["macd_hist"].iloc[-1]
        return f"MACD histogram {hist * 10_000:+.2f} bps ({'building' if value > 0 else 'fading'})"


class RocMomentumStrategy(Strategy):
    name = "roc_momentum"
    category = "momentum"
    description = "Multi-horizon rate-of-change agreement"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        horizons = [f"roc_{window}" for window in (5, 10, 20) if f"roc_{window}" in features.columns]
        if not horizons:
            return self._empty(context)

        # All three horizons must agree for the signal to be strong.
        agreement = sum(np.sign(features[h]) for h in horizons) / len(horizons)
        magnitude = features[horizons].abs().mean(axis=1)
        return squash(agreement * magnitude, scale=0.0025).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Multi-horizon momentum {'up' if value > 0 else 'down'}"


class StochasticReversalStrategy(Strategy):
    name = "stochastic"
    category = "momentum"
    description = "Stochastic %K/%D crossover at range extremes"

    def __init__(self, oversold: float = 20.0, overbought: float = 80.0, hold_bars: int = 5) -> None:
        self.oversold = oversold
        self.overbought = overbought
        self.hold_bars = hold_bars

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["stoch_k", "stoch_cross"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        k = features["stoch_k"]
        cross = features["stoch_cross"].fillna(0.0)

        # A crossover is a one-bar event, so hold it briefly to be tradeable.
        bullish = ((cross > 0) & (k < self.oversold)).astype(float)
        bearish = ((cross < 0) & (k > self.overbought)).astype(float)
        bullish = bullish.rolling(self.hold_bars, min_periods=1).max()
        bearish = bearish.rolling(self.hold_bars, min_periods=1).max()

        # Deeper into the extreme means a stronger reversal case.
        depth_up = ((self.oversold - k) / self.oversold).clip(0.0, 1.0)
        depth_down = ((k - self.overbought) / (100.0 - self.overbought)).clip(0.0, 1.0)

        raw = bullish * (0.4 + 0.6 * depth_up) - bearish * (0.4 + 0.6 * depth_down)
        return squash(raw, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        k = context.features["stoch_k"].iloc[-1]
        return f"Stochastic %K {k:.0f}"


class ActivitySpikeStrategy(Strategy):
    name = "activity_spike"
    category = "momentum"
    description = "Directional move on unusual activity (volume, or tick count for an index)"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "activity_z_20" not in features.columns:
            return self._empty(context)

        # A move backed by abnormal participation is more likely to continue.
        spike = ((features["activity_z_20"] - 0.5) / 1.5).clip(0.0, 1.0)
        direction = np.sign(features["ret_5"].fillna(0.0))
        return squash(direction * spike, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        z = context.features["activity_z_20"].iloc[-1]
        return f"Activity {z:+.1f} sigma with {'up' if value > 0 else 'down'} bias"


class OrderFlowImbalanceStrategy(Strategy):
    """Order-flow imbalance from futures depth.

    Requires a futures instrument — the NIFTY 50 index itself publishes no order
    book, so this returns a flat series unless depth data was fed in. Returning
    zeros is the right behaviour: it lets the ensemble include the strategy for
    instruments that support it without pretending it has an opinion when the
    inputs are absent.
    """

    name = "order_flow"
    category = "flow"
    description = "Order-book and traded-value imbalance from futures depth"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        candidates = ["depth_imbalance", "trade_imbalance", "depth_imbalance_ma"]
        available = [column for column in candidates if column in features.columns]
        if not available:
            return self._empty(context)

        blended = sum(features[column].fillna(0.0) for column in available) / len(available)
        return squash(blended, scale=0.5).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Order flow {'bid' if value > 0 else 'ask'} heavy"


# The classes this package publishes, in one place: the plugin manifest derives
# its provided capabilities from this tuple, so a strategy cannot exist in code
# without being advertised to the kernel.

# The classes this pack publishes, in one place. The plugin manifest derives its
# provided capabilities from this tuple, so a strategy cannot exist in code without
# being advertised to the kernel — and cannot be advertised without existing.
STRATEGIES = (
    MacdMomentumStrategy,
    RocMomentumStrategy,
    StochasticReversalStrategy,
    ActivitySpikeStrategy,
    OrderFlowImbalanceStrategy,
)
