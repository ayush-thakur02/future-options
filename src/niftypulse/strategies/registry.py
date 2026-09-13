"""Strategy registry and the ML-backed strategy.

The ML strategy is defined here rather than in ``ml/`` to keep the dependency
one-directional: strategies may consume a predictor, but the ML package never
needs to know that strategies exist. The predictor is injected as a callable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..ml.metrics import apply_expected_move_curve
from .base import CompositeStrategy, Strategy, StrategyContext, squash

PredictorFn = Callable[[pd.DataFrame], np.ndarray]


class MLStrategy(Strategy):
    """Turns model probabilities into a signed conviction series.

    Conviction is ``2p - 1`` scaled by a confidence gate, so a 0.52 probability
    produces a small signal rather than a full-size one. Treating every positive
    prediction as a full-strength trade is the most common way a mediocre model
    gets turned into a losing strategy.

    On a scalping timescale there is a second gate that matters more than the
    first. A forecast can be directionally right and still lose money if the move
    it predicts is smaller than the round-trip cost, so when a fitted
    conviction-to-move curve is supplied the strategy stays silent unless the
    expected move clears the hurdle. Most bars will not clear it. That is the
    point.
    """

    name = "ml"
    category = "ml"
    description = "Gradient-boosted ensemble direction forecast"

    def __init__(
        self,
        predictor: PredictorFn,
        feature_names: list[str] | None = None,
        min_edge: float = 0.04,
        move_curve: dict | None = None,
        hurdle_bps: float = 0.0,
    ) -> None:
        self.predictor = predictor
        self.feature_names = feature_names
        self.min_edge = min_edge
        self.move_curve = move_curve or {}
        self.hurdle_bps = float(hurdle_bps)

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        columns = self.feature_names or list(features.columns)
        present = [column for column in columns if column in features.columns]
        if not present:
            return self._empty(context)

        matrix = features[present]
        try:
            probabilities = np.asarray(self.predictor(matrix), dtype="float64")
        except Exception:
            return self._empty(context)
        if probabilities.shape[0] != len(matrix):
            return self._empty(context)

        edge = 2.0 * probabilities - 1.0
        # Below min_edge the model is within noise; treat it as no opinion.
        gated = np.where(np.abs(edge) >= self.min_edge, edge, 0.0)

        # Cost gate: stay flat unless the move this conviction historically
        # preceded is large enough to cover the round trip.
        if self.move_curve and self.hurdle_bps > 0:
            expected_moves = np.array(
                [apply_expected_move_curve(abs(value), self.move_curve) for value in gated],
                dtype="float64",
            )
            gated = np.where(expected_moves > self.hurdle_bps, gated, 0.0)

        scaled = np.sign(gated) * (np.abs(gated) - self.min_edge) / (1.0 - self.min_edge)
        return pd.Series(scaled, index=features.index, dtype="float64").clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"ML model implies {'up' if value > 0 else 'down'}"


class HybridStrategy(Strategy):
    """Blend of rule-based conviction and model probability.

    Rules are legible but brittle; models adapt but are opaque. Blending them
    with a configurable weight keeps a human-readable sanity check in the loop —
    when the two disagree strongly, that disagreement is itself informative.
    """

    name = "hybrid"
    category = "ensemble"
    description = "Rule-based ensemble blended with the ML forecast"

    def __init__(
        self,
        rules: CompositeStrategy,
        model: MLStrategy,
        model_weight: float = 0.55,
    ) -> None:
        self.rules = rules
        self.model = model
        self.model_weight = float(np.clip(model_weight, 0.0, 1.0))

    def score(self, context: StrategyContext) -> pd.Series:
        rule_score = self.rules.score(context).fillna(0.0)
        model_score = self.model.score(context).fillna(0.0)
        blended = (1.0 - self.model_weight) * rule_score + self.model_weight * model_score
        return squash(blended, scale=0.9).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Hybrid view {'up' if value > 0 else 'down'}"


def _registry() -> dict[str, type[Strategy]]:
    from .momentum import (
        ActivitySpikeStrategy,
        MacdMomentumStrategy,
        OrderFlowImbalanceStrategy,
        StochasticReversalStrategy,
    )
    from .reversion import (
        BollingerReversionStrategy,
        RangeFadeStrategy,
        RsiReversionStrategy,
        VwapReversionStrategy,
        ZScoreReversionStrategy,
    )
    from .trend import (
        DonchianBreakoutStrategy,
        EfficiencyTrendStrategy,
        EmaTrendStrategy,
        OpeningRangeBreakoutStrategy,
        SuperTrendStrategy,
    )
    from .volatility import (
        GarchVolForecastStrategy,
        SqueezeReleaseStrategy,
        VolatilityBreakoutStrategy,
        VolatilityMeanReversionStrategy,
    )

    classes: list[type[Strategy]] = [
        EmaTrendStrategy,
        SuperTrendStrategy,
        DonchianBreakoutStrategy,
        OpeningRangeBreakoutStrategy,
        EfficiencyTrendStrategy,
        RsiReversionStrategy,
        BollingerReversionStrategy,
        VwapReversionStrategy,
        ZScoreReversionStrategy,
        RangeFadeStrategy,
        MacdMomentumStrategy,
        StochasticReversalStrategy,
        ActivitySpikeStrategy,
        OrderFlowImbalanceStrategy,
        SqueezeReleaseStrategy,
        VolatilityBreakoutStrategy,
        VolatilityMeanReversionStrategy,
        GarchVolForecastStrategy,
    ]
    return {cls.name: cls for cls in classes}


STRATEGY_REGISTRY: dict[str, type[Strategy]] = _registry()


# Prior weights for the default ensemble. These encode a view about what works
# on intraday index data — trend and flow carry more weight than oscillator
# fades — and are meant to be replaced by measured weights after a backtest run.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ema_trend": 1.0,
    "supertrend": 0.9,
    "efficiency_trend": 0.8,
    "donchian_breakout": 0.7,
    "orb": 0.6,
    "macd_momentum": 0.8,
    "roc_momentum": 0.7,
    "squeeze_release": 0.7,
    "vol_breakout": 0.6,
    "vol_regime": 0.5,
    "rsi_reversion": 0.7,
    "bollinger_reversion": 0.7,
    "vwap_reversion": 0.6,
    "zscore_reversion": 0.5,
    "range_fade": 0.4,
    "stochastic": 0.4,
    "activity_spike": 0.5,
    "order_flow": 0.9,
    "vol_reversion": 0.4,
}


def available_strategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY)


def build_strategy(name: str, **kwargs) -> Strategy:
    if name == "ensemble":
        return default_ensemble(**kwargs)
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(available_strategies())}")
    return STRATEGY_REGISTRY[name](**kwargs)


def default_ensemble(weights: dict[str, float] | None = None) -> CompositeStrategy:
    """Build the composite strategy from the registry and a weight map."""
    weights = weights or DEFAULT_WEIGHTS
    ensemble = CompositeStrategy()
    for name, weight in weights.items():
        if name in STRATEGY_REGISTRY and weight:
            ensemble.add(STRATEGY_REGISTRY[name](), weight)
    return ensemble


def all_strategies() -> list[Strategy]:
    return [cls() for cls in STRATEGY_REGISTRY.values()]
