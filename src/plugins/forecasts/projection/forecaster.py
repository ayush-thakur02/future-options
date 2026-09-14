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

from core.calendar import IST, SESSION_CLOSE, SESSION_OPEN
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
        learned_closes: dict[int, float] | None = None,
        record: bool = True,
        issued_at=None,
        context: dict | None = None,
        session_only: bool = False,
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
        if session_only:
            self.projections = [c for c in self.projections if SESSION_OPEN <= c.ts.time() < SESSION_CLOSE]
        # Rule/tape estimates retain at least 75% of the path. The separately
        # trained instrument model contributes only after sufficient samples.
        for candle in self.projections:
            if learned_closes and candle.horizon in learned_closes:
                shifted_close = 0.75 * candle.close + 0.25 * learned_closes[candle.horizon]
                candle.close = max(shifted_close, 0.01)
                candle.open = self.anchor if candle.horizon == 1 else self.projections[candle.horizon - 2].close
                candle.high = max(candle.high, candle.open, candle.close)
                candle.low = max(min(candle.low, candle.open, candle.close), 0.0)
                candle.expected_move_bps = candle.change_bps
        self.updated_at = (now or datetime.now(IST)).astimezone(IST)
        self.updates += 1
        if record:
            self.tracker.record(self.projections, self.anchor, issued_at=issued_at, context=context)
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
