"""Volatility-regime strategies.

Volatility is the one input that genuinely forecasts itself: quiet periods
cluster and then resolve into expansion. These strategies trade that transition
rather than predicting direction from volatility alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, squash


class SqueezeReleaseStrategy(Strategy):
    name = "squeeze_release"
    category = "volatility"
    description = "Trade the expansion after a Bollinger squeeze, in the breakout's direction"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["bb_squeeze", "keltner_position", "ret_5"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        squeeze = features["bb_squeeze"]
        # Was compressed recently, and is now expanding.
        was_tight = (squeeze.rolling(20, min_periods=5).min() < 0.8).astype(float)
        expanding = (squeeze.diff(3) > 0).astype(float)
        release = was_tight * expanding

        direction = np.sign(features["ret_5"].fillna(0.0))
        # Keltner position disambiguates which edge is being broken.
        edge = np.sign(features["keltner_position"].fillna(0.0))
        agree = np.sign(direction + edge)
        return squash(agree * release, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        squeeze = context.features["bb_squeeze"].iloc[-1]
        return f"Squeeze release (bandwidth {squeeze:.2f}x baseline), {'up' if value > 0 else 'down'}"


class VolatilityBreakoutStrategy(Strategy):
    name = "vol_breakout"
    category = "volatility"
    description = "Range expansion beyond ATR, in the direction of the break"

    def __init__(self, atr_multiple: float = 1.1) -> None:
        self.atr_multiple = atr_multiple

    def score(self, context: StrategyContext) -> pd.Series:
        bars = context.bars
        features = context.features
        if "atr_norm" not in features.columns or bars.empty:
            return self._empty(context)

        bar_range = (bars["high"] - bars["low"]) / bars["close"].replace(0, np.nan)
        atr_norm = features["atr_norm"]
        expansion = (bar_range / (atr_norm + 1e-12)) - 1.0

        triggered = (expansion > self.atr_multiple).astype(float)
        direction = np.sign(bars["close"] - bars["open"])
        # Conviction scales with how far past the threshold the expansion ran.
        magnitude = ((expansion - self.atr_multiple) / 1.5).clip(0.0, 1.0)
        return squash(direction * triggered * (0.4 + 0.6 * magnitude), scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"ATR range expansion {'up' if value > 0 else 'down'}"


class VolatilityMeanReversionStrategy(Strategy):
    """Trade the unwind of a volatility spike.

    After an outsized bar the next few bars tend to be quieter and partially
    retrace. This is the opposite bet to the breakout strategy, which is what
    makes the two useful together.
    """

    name = "vol_reversion"
    category = "volatility"
    description = "Fade an outsized candle once volatility starts to subside"

    def score(self, context: StrategyContext) -> pd.Series:
        bars = context.bars
        features = context.features
        if "atr_ratio" not in features.columns or bars.empty:
            return self._empty(context)

        # Volatility elevated but already rolling over.
        elevated = ((features["atr_ratio"] - 1.4) / 1.0).clip(0.0, 1.0)
        cooling = (features["atr_ratio"].diff(3) < 0).astype(float)

        candle = np.sign(bars["close"] - bars["open"])
        # Fade the direction of the extreme move.
        return squash(-candle * elevated * cooling, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        ratio = context.features["atr_ratio"].iloc[-1]
        return f"Volatility {ratio:.2f}x baseline, fading the spike"


class GarchVolForecastStrategy(Strategy):
    """Directionless volatility signal used as a position-size input.

    It emits conviction proportional to how far current volatility sits from its
    own recent norm. Direction comes from the trend stack, since volatility alone
    says nothing about which way price goes.
    """

    name = "vol_regime"
    category = "volatility"
    description = "Volatility regime tilt combined with trend direction"

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        needed = ["vol_ratio_10_60", "supertrend_dir", "efficiency_ratio_10"]
        if not all(column in features.columns for column in needed):
            return self._empty(context)

        # Expanding volatility amplifies a clean trend; contracting damps it.
        expanding = squash(features["vol_ratio_10_60"] - 1.0, scale=0.5)
        cleanliness = ((features["efficiency_ratio_10"] - 0.2) / 0.5).clip(0.0, 1.0)
        raw = features["supertrend_dir"].fillna(0.0) * (0.5 + 0.5 * expanding) * cleanliness
        return squash(raw, scale=0.7).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        ratio = context.features["vol_ratio_10_60"].iloc[-1]
        return f"Vol regime {ratio:.2f}x, trend {'up' if value > 0 else 'down'}"


# The classes this package publishes, in one place: the plugin manifest derives
# its provided capabilities from this tuple, so a strategy cannot exist in code
# without being advertised to the kernel.

# The classes this pack publishes, in one place. The plugin manifest derives its
# provided capabilities from this tuple, so a strategy cannot exist in code without
# being advertised to the kernel — and cannot be advertised without existing.
STRATEGIES = (
    SqueezeReleaseStrategy,
    VolatilityBreakoutStrategy,
    VolatilityMeanReversionStrategy,
    GarchVolForecastStrategy,
)
