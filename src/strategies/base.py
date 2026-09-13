"""Strategy interface.

Strategies are **vectorised**: given the full bar and feature frames they return
a signed conviction series in [-1, 1] for every bar at once. This is a deliberate
departure from the event-driven style common in trading frameworks.

The reason is that a rule evaluated per-bar in a loop cannot be validated until
it has been looped over, which makes research slow and mistakes expensive. With a
vectorised form the backtest is a single pass over a Series, live inference is
just ``.iloc[-1]``, and the same code produces both — so a strategy that looks
good in research is literally the strategy running live, not a re-implementation
of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

import numpy as np
import pandas as pd

from core.types import Direction, Signal


@dataclass
class StrategyContext:
    """Inputs a strategy may use."""

    bars: pd.DataFrame
    features: pd.DataFrame
    extras: dict = field(default_factory=dict)


class Strategy(ABC):
    """Base class for all rule-based strategies."""

    name: ClassVar[str] = "strategy"
    category: ClassVar[str] = "general"
    description: ClassVar[str] = ""

    @abstractmethod
    def score(self, context: StrategyContext) -> pd.Series:
        """Signed conviction in [-1, 1] for every bar."""

    # ----------------------------------------------------------------- helpers

    def _empty(self, context: StrategyContext) -> pd.Series:
        return pd.Series(0.0, index=context.features.index, dtype="float64")

    def _latest_signal(self, context: StrategyContext, threshold: float = 0.15) -> Signal | None:
        series = self.score(context).dropna()
        if series.empty:
            return None
        value = float(series.iloc[-1])
        if abs(value) < threshold:
            return None
        return Signal(
            ts=_last_timestamp(series.index),
            strategy=self.name,
            direction=Direction.UP if value > 0 else Direction.DOWN,
            strength=min(abs(value), 1.0),
            reason=self.describe(value, context),
        )

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"{self.name} score {value:+.2f}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def _last_timestamp(index: pd.Index) -> datetime:
    stamp = index[-1]
    return stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else datetime.now()


def squash(series: pd.Series, scale: float = 1.0, clip: float = 1.0) -> pd.Series:
    """Map an unbounded score into [-clip, clip] with a smooth saturating curve."""
    scaled = series / (scale + 1e-12)
    return clip * np.tanh(scaled)


def gaussian_bell(series: pd.Series, center: float, width: float) -> pd.Series:
    """Peak of 1 at ``center``, decaying with ``width``; used for threshold bands."""
    return np.exp(-(((series - center) / (width + 1e-12)) ** 2))


def require_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in frame.columns for column in columns)


class CompositeStrategy(Strategy):
    """Weighted blend of several strategies, blending their signed scores."""

    name = "composite"
    category = "ensemble"
    description = "Weighted blend of component strategies"

    def __init__(self, components: list[tuple[Strategy, float]] | None = None) -> None:
        self.components: list[tuple[Strategy, float]] = components or []

    def add(self, strategy: Strategy, weight: float = 1.0) -> CompositeStrategy:
        self.components.append((strategy, weight))
        return self

    def score(self, context: StrategyContext) -> pd.Series:
        if not self.components:
            return self._empty(context)

        total_weight = sum(abs(weight) for _, weight in self.components) or 1.0
        blended = pd.Series(0.0, index=context.features.index, dtype="float64")
        for strategy, weight in self.components:
            blended = blended + strategy.score(context).fillna(0.0) * weight
        return (blended / total_weight).clip(-1.0, 1.0)
