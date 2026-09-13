"""The projection forecaster: what the runtime drives once a second.

It holds three things that belong together and nowhere else:

* the last projection, so a reader can ask for it without recomputing
* :class:`MicroMomentum`, which turns the last few seconds of tape into a
  conviction contribution
* :class:`ProjectionTracker`, which scores every projection against the bar that
  eventually closes

The forecaster deliberately does not decide *when* to refresh. The runtime owns
that cadence, so the same object serves a one-second live loop and a fast replay
without a mode switch inside it.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from core.calendar import IST
from core.types import ForecastCandle, Tick

from .candles import DEFAULT_BARS_AHEAD, MICRO_WEIGHT, atr_of, project_candles
from .momentum import MicroMomentum
from .tracker import ProjectionTracker


class ProjectionForecaster:
    """Projects the next few candles from live price and current conviction."""

    def __init__(
        self,
        bars_ahead: int = DEFAULT_BARS_AHEAD,
        micro_weight: float = MICRO_WEIGHT,
        tracker: ProjectionTracker | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.bars_ahead = int(max(bars_ahead, 1))
        self.micro_weight = float(np.clip(micro_weight, 0.0, 1.0))
        self.momentum = MicroMomentum()
        self.tracker = tracker or ProjectionTracker()
        self.log = logger or logging.getLogger("niftypulse.projection")
        self.projections: list[ForecastCandle] = []
        self.conviction: float = 0.0
        self.micro: float = 0.0
        self.atr: float = 0.0
        self.anchor: float = 0.0
        self.updated_at: datetime | None = None
        self.updates: int = 0

    # ----------------------------------------------------------------- ticks

    def on_tick(self, tick: Tick) -> None:
        """Fold a tick into the momentum estimate. Cheap, and called per tick."""
        self.momentum.update(tick)

    # -------------------------------------------------------------- projection

    def refresh(
        self,
        bars: pd.DataFrame,
        anchor: float,
        conviction: float,
        now: datetime | None = None,
        bar_minutes: int = 1,
    ) -> list[ForecastCandle]:
        """Rebuild the projected path, and record it for scoring.

        ``conviction`` is the slower view — strategies blended with any model
        forecast. The live tick momentum is blended in here rather than by the
        caller, because how much the last twenty seconds should override a
        one-minute view is a property of the projection, not of the caller.
        """
        atr = atr_of(bars)
        self.atr = atr
        self.anchor = float(anchor)
        self.conviction = float(np.clip(conviction, -1.0, 1.0))
        self.micro = float(self.momentum.value(atr)) if atr > 0 else 0.0

        blended = (1.0 - self.micro_weight) * self.conviction + self.micro_weight * self.micro
        blended = float(np.clip(blended, -1.0, 1.0))

        self.projections = project_candles(
            bars=bars,
            anchor=self.anchor,
            conviction=blended,
            now=now,
            bars_ahead=self.bars_ahead,
            bar_minutes=bar_minutes,
        )
        self.updated_at = (now or datetime.now(IST)).astimezone(IST)
        self.updates += 1
        self.tracker.record(self.projections, self.anchor)
        return self.projections

    # --------------------------------------------------------------- scoring

    def observe_bar(self, bar_ts, close: float) -> None:
        """Score any projection that targeted the bar that just closed."""
        self.tracker.observe(bar_ts, close)

    # ------------------------------------------------------------ diagnostics

    @property
    def age_seconds(self) -> float:
        if self.updated_at is None:
            return float("inf")
        return (datetime.now(IST) - self.updated_at).total_seconds()

    @property
    def summary(self) -> str:
        if not self.projections:
            return "no projection"
        first, last = self.projections[0], self.projections[-1]
        total_bps = (last.close / first.open - 1.0) * 10_000 if first.open else 0.0
        return f"{'UP' if total_bps >= 0 else 'DOWN'} {total_bps:+.1f}bp over {len(self.projections)} bars"

    def describe(self) -> dict:
        return {
            "bars_ahead": self.bars_ahead,
            "conviction": round(self.conviction, 4),
            "micro": round(self.micro, 4),
            "atr": round(self.atr, 2),
            "updates": self.updates,
            "tracking": self.tracker.summary(),
            "per_horizon": {h: s.as_row() for h, s in self.tracker.stats().items()},
        }

    def __repr__(self) -> str:
        return f"<ProjectionForecaster {self.bars_ahead} bars, {self.updates} refreshes>"


__all__ = ["ProjectionForecaster"]
