"""Trend-following strategies.

Trend systems work when the market is directional and bleed when it chops, so
every strategy here gates on a trend-quality measure (ADX, efficiency ratio, or
the trend's own persistence) rather than emitting a signal unconditionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyContext, squash


class EmaTrendStrategy(Strategy):
    name = "ema_trend"
    category = "trend"
    description = "EMA 9/21 alignment confirmed by ADX trend strength"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["ema_dist_9", "ema_dist_21", "ema_dist_50", "adx_14", "di_spread"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        # Stacked EMAs give a graded signal rather than a binary cross.
        alignment = (
            np.sign(features["ema_dist_9"] - features["ema_dist_21"])
            + np.sign(features["ema_dist_21"] - features["ema_dist_50"])
            + np.sign(features["ema_dist_9"] - features["ema_dist_50"])
        ) / 3.0

        strength = features["di_spread"].abs() / 30.0
        adx_gate = ((features["adx_14"] - 18.0) / 15.0).clip(0.0, 1.0)

        raw = alignment * strength * adx_gate
        return squash(raw, scale=0.6).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        adx = context.features["adx_14"].iloc[-1]
        return f"EMA stack {'bullish' if value > 0 else 'bearish'}, ADX {adx:.0f}"


class SuperTrendStrategy(Strategy):
    name = "supertrend"
    category = "trend"
    description = "ATR-based SuperTrend direction with distance-scaled conviction"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "supertrend_dir" not in features.columns:
            return self._empty(context)

        distance = features.get("supertrend_dist", pd.Series(0.0, index=features.index))
        direction = features["supertrend_dir"].fillna(0.0)

        # Persistence: how many bars the current trend has held.
        flips = direction.ne(direction.shift(1)).cumsum()
        run_length = direction.groupby(flips).cumcount()
        maturity = (run_length.clip(upper=20) / 20.0) * 0.5 + 0.5

        # SuperTrend is always either long or short, so without a quality gate it
        # would hold a strong opinion in dead chop and dominate any blend it sits
        # in. Gate on trend quality so it stays quiet when there is no trend.
        quality = pd.Series(1.0, index=features.index, dtype="float64")
        if "efficiency_ratio_10" in features.columns:
            quality = ((features["efficiency_ratio_10"] - 0.15) / 0.4).clip(0.0, 1.0)
        if "adx_14" in features.columns:
            adx_gate = ((features["adx_14"] - 15.0) / 15.0).clip(0.0, 1.0)
            quality = np.minimum(quality, adx_gate)

        conviction = direction * maturity * quality * (1.0 + squash(distance.abs(), scale=0.004))
        return squash(conviction, scale=0.8).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"SuperTrend {'up' if value > 0 else 'down'}"


class DonchianBreakoutStrategy(Strategy):
    name = "donchian_breakout"
    category = "trend"
    description = "20-bar channel breakout, filtered by Bollinger squeeze release"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["breakout_high_20", "breakout_low_20", "donchian_position", "bb_squeeze"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        breakout = features["breakout_high_20"].fillna(0.0) - features["breakout_low_20"].fillna(0.0)
        # Squeeze release is what makes a breakout worth taking.
        squeeze_gate = (1.0 / (1.0 + features["bb_squeeze"].clip(lower=0.1)))
        return squash(breakout * squeeze_gate * 1.5, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"20-bar breakout {'up' if value > 0 else 'down'}"


class OpeningRangeBreakoutStrategy(Strategy):
    name = "orb"
    category = "trend"
    description = "Break of the first-15-minute range, active until midday"

    def score(self, context: StrategyContext) -> pd.Series:
        bars = context.bars
        features = context.features
        if "minutes_from_open" not in features.columns or bars.empty:
            return self._empty(context)

        day = pd.Series(bars.index.normalize(), index=bars.index)
        minutes = features["minutes_from_open"]

        # Range of the first 15 minutes, broadcast across the session.
        opening = bars["high"].where(minutes <= 15).groupby(day).cummax()
        opening_low = bars["low"].where(minutes <= 15).groupby(day).cummin()
        opening_high = opening.groupby(day).transform("max")
        opening_low = opening_low.groupby(day).transform("min")

        span = (opening_high - opening_low).replace(0, np.nan)
        position = (bars["close"] - opening_high) / (span + 1e-12)
        downside = (opening_low - bars["close"]) / (span + 1e-12)
        raw = position.fillna(0.0) + (-downside.fillna(0.0))

        # Only trade the window where ORB has any edge, and only after the range forms.
        window = ((minutes > 15) & (minutes < 240)).astype(float)
        return squash(raw * window, scale=0.35).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Opening-range break {'up' if value > 0 else 'down'}"


class EfficiencyTrendStrategy(Strategy):
    name = "efficiency_trend"
    category = "trend"
    description = "Kaufman efficiency ratio: follows only clean, low-noise moves"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["efficiency_ratio_10", "slope_20", "slope_50"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        # Efficiency ratio near 1 means the move travelled in a straight line.
        cleanliness = ((features["efficiency_ratio_10"] - 0.2) / 0.5).clip(0.0, 1.0)
        direction = np.sign(features["slope_20"].fillna(0.0) + features["slope_50"].fillna(0.0))
        return squash(direction * cleanliness, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        ratio = context.features["efficiency_ratio_10"].iloc[-1]
        return f"Trend efficiency {ratio:.2f}"
