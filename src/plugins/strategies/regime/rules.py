"""Rules that change their interpretation when the market state changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, require_columns, squash

EPS = 1e-12


class HurstAdaptiveStrategy(Strategy):
    name = "hurst_adaptive"
    category = "regime"
    description = "Follow persistent markets and fade anti-persistent markets using the Hurst exponent"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["hurst_100", "ret_5", "ret_20", "price_z_100"]
        if not require_columns(features, needed):
            return self._empty(context)

        hurst = features["hurst_100"]
        persistence = ((hurst - 0.52) / 0.16).clip(0.0, 1.0)
        anti_persistence = ((0.48 - hurst) / 0.16).clip(0.0, 1.0)
        trend = squash(features["ret_20"], scale=0.005)
        reversion = squash(-features["price_z_100"], scale=2.0)
        raw = persistence * trend + anti_persistence * reversion
        return squash(raw, scale=0.65).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        hurst = float(context.features["hurst_100"].iloc[-1])
        mode = "persistent" if hurst >= 0.5 else "mean-reverting"
        return f"Hurst {hurst:.2f} ({mode})"


class DirectionalEntropyStrategy(Strategy):
    name = "directional_entropy"
    category = "regime"
    description = "Follow predictable low-entropy return signs; abstain when direction is random"

    def __init__(self, window: int = 24) -> None:
        self.window = window

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if "ret_1" not in features.columns:
            return self._empty(context)

        up = (features["ret_1"] > 0).astype("float64")
        probability = up.rolling(self.window, min_periods=self.window).mean().clip(EPS, 1.0 - EPS)
        entropy = -(probability * np.log2(probability) + (1.0 - probability) * np.log2(1.0 - probability))
        predictability = (1.0 - entropy).clip(0.0, 1.0)
        direction = ((probability - 0.5) * 2.0).clip(-1.0, 1.0)

        # Efficiency rejects low-entropy sequences caused by tiny alternating
        # prints whose net path is still noise.
        efficiency = features.get(
            "efficiency_ratio_10", pd.Series(1.0, index=features.index)
        ).clip(0.0, 1.0)
        raw = direction * predictability * (0.25 + 0.75 * efficiency)
        return squash(raw, scale=0.20).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Low directional entropy favors {'up' if value > 0 else 'down'} persistence"


class VolatilityStateRotationStrategy(Strategy):
    name = "volatility_state_rotation"
    category = "regime"
    description = "Follow momentum as volatility expands and fade z-score extension as it contracts"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["vol_ratio_10_60", "ret_10", "ret_z_20"]
        if not require_columns(features, needed):
            return self._empty(context)

        ratio = features["vol_ratio_10_60"]
        expansion = ((ratio - 1.0) / 0.65).clip(0.0, 1.0)
        contraction = ((1.0 - ratio) / 0.45).clip(0.0, 1.0)
        momentum = squash(features["ret_10"], scale=0.0035)
        fade = squash(-features["ret_z_20"], scale=1.6)
        raw = expansion * momentum + contraction * fade
        return squash(raw, scale=0.65).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        ratio = float(context.features["vol_ratio_10_60"].iloc[-1])
        mode = "expansion" if ratio >= 1.0 else "contraction"
        return f"Volatility {mode} at {ratio:.2f}x"


STRATEGIES = (
    HurstAdaptiveStrategy,
    DirectionalEntropyStrategy,
    VolatilityStateRotationStrategy,
)
