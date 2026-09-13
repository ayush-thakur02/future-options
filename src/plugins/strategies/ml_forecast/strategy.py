"""The trained model as a strategy.

Two gates stand between a model probability and a tradeable conviction, and both
matter more than the model does on a scalping timescale.

The first is **edge**: a probability of 0.502 is noise, so anything below
``min_edge`` produces no opinion at all rather than a very small one.

The second is **cost**. A forecast can be directionally right and still lose
money, because the move it predicts may be smaller than the round trip needed to
capture it. When a fitted conviction-to-move curve is available the strategy stays
silent unless the expected move clears the hurdle. Most bars will not clear it.
That is the point, and it is why the conviction series is mostly zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.forecasts.ml_ensemble.metrics import apply_expected_move_curve

from ..base import Strategy, StrategyContext


class MLStrategy(Strategy):
    """Model probabilities turned into a gated, signed conviction series."""

    name = "ml"
    category = "ml"
    description = "Cost-gated conviction from the trained direction model"

    def __init__(
        self,
        predictor,
        horizon: int = 1,
        feature_names: list[str] | None = None,
        min_edge: float = 0.04,
        move_curve: dict | None = None,
        hurdle_bps: float = 0.0,
    ) -> None:
        self.predictor = predictor
        self.horizon = int(horizon)
        self.feature_names = feature_names
        self.min_edge = float(min_edge)
        self.move_curve = move_curve or {}
        self.hurdle_bps = float(hurdle_bps)

    def score(self, context: StrategyContext) -> pd.Series:
        features = context.features
        if features.empty:
            return self._empty(context)

        # Inference goes through the predictor's own vectorised path, so the
        # series this produces is the same one the live engine acts on.
        try:
            probabilities = self.predictor.predict_frame(features, self.horizon)
        except Exception:  # noqa: BLE001 — a model that cannot score means no opinion
            return self._empty(context)
        if probabilities.empty:
            return self._empty(context)

        probabilities = probabilities.reindex(features.index)
        edge = 2.0 * probabilities.to_numpy(dtype="float64") - 1.0
        gated = np.where(np.abs(edge) >= self.min_edge, edge, 0.0)

        if self.move_curve and self.hurdle_bps > 0:
            expected = np.array(
                [apply_expected_move_curve(abs(value), self.move_curve) for value in gated],
                dtype="float64",
            )
            gated = np.where(expected > self.hurdle_bps, gated, 0.0)

        scaled = np.sign(gated) * (np.abs(gated) - self.min_edge) / (1.0 - self.min_edge)
        return pd.Series(np.nan_to_num(scaled), index=features.index, dtype="float64").clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"ML model implies {'up' if value > 0 else 'down'}"

    def __repr__(self) -> str:
        return f"<MLStrategy horizon={self.horizon}m min_edge={self.min_edge}>"


__all__ = ["MLStrategy"]
