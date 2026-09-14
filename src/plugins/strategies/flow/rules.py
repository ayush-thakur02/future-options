"""Strategies based on participation and the location of traded activity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, require_columns, squash


class ChaikinFlowTrendStrategy(Strategy):
    name = "chaikin_flow_trend"
    category = "flow"
    description = "Directional momentum confirmed by Chaikin money flow and participation"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["cmf_20", "ret_5", "activity_ratio_20"]
        if not require_columns(features, needed):
            return self._empty(context)

        flow = squash(features["cmf_20"], scale=0.16)
        momentum = squash(features["ret_5"], scale=0.0025)
        agrees = (np.sign(flow) == np.sign(momentum)).astype("float64")
        participation = ((features["activity_ratio_20"] - 0.75) / 0.75).clip(0.0, 1.0)
        raw = (0.55 * flow + 0.45 * momentum) * agrees * (0.4 + 0.6 * participation)
        return squash(raw, scale=0.65).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        flow = float(context.features["cmf_20"].iloc[-1])
        return f"Chaikin flow {flow:+.2f}, {'buying' if value > 0 else 'selling'} pressure"


class MoneyFlowReversalStrategy(Strategy):
    name = "money_flow_reversal"
    category = "flow"
    description = "MFI extreme turning back with candle-location confirmation"

    def __init__(self, low: float = 22.0, high: float = 78.0) -> None:
        self.low = low
        self.high = high

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "mfi_14" not in features.columns:
            return self._empty(context)

        mfi = features["mfi_14"]
        slope = mfi.diff(3)
        bullish = ((mfi < self.low) & (slope > 0)).astype("float64")
        bearish = ((mfi > self.high) & (slope < 0)).astype("float64")
        up_depth = ((self.low - mfi) / self.low).clip(0.0, 1.0)
        down_depth = ((mfi - self.high) / (100.0 - self.high)).clip(0.0, 1.0)

        candle = features.get("close_location", pd.Series(0.5, index=features.index))
        up_confirm = (0.5 + candle.clip(0.0, 1.0)) / 1.5
        down_confirm = (1.5 - candle.clip(0.0, 1.0)) / 1.5
        raw = bullish * (0.45 + 0.55 * up_depth) * up_confirm
        raw -= bearish * (0.45 + 0.55 * down_depth) * down_confirm
        return squash(raw, scale=0.65).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        mfi = float(context.features["mfi_14"].iloc[-1])
        return f"Money Flow Index {mfi:.0f} turning {'up' if value > 0 else 'down'}"


class ObvDivergenceStrategy(Strategy):
    name = "obv_divergence"
    category = "flow"
    description = "Price/OBV slope divergence with an activity gate"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["obv_slope_20", "slope_20", "activity_ratio_20"]
        if not require_columns(features, needed):
            return self._empty(context)

        price_slope = squash(features["slope_20"], scale=0.00008)
        obv_slope = squash(features["obv_slope_20"], scale=0.08)
        divergent = (np.sign(price_slope) != np.sign(obv_slope)).astype("float64")
        # OBV leading price is the direction of the anticipated convergence.
        separation = (obv_slope - price_slope) / 2.0
        participation = ((features["activity_ratio_20"] - 0.8) / 0.7).clip(0.0, 1.0)
        raw = separation * divergent * (0.3 + 0.7 * participation)
        return squash(raw, scale=0.45).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"OBV leads price {'higher' if value > 0 else 'lower'}"


STRATEGIES = (
    ChaikinFlowTrendStrategy,
    MoneyFlowReversalStrategy,
    ObvDivergenceStrategy,
)
