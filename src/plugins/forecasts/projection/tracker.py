"""Scoring projections against what actually happened.

A projected candle that nobody checks is decoration. This records each projection
against its target bar and scores it when that bar closes: did the path lean the
right way, and by how much did the close miss?

Scores are kept **per horizon**, because the three horizons answer different
questions. A one-bar-ahead projection is nearly a nowcast — the forming bar
already contains most of it. A three-bar-ahead projection is a real claim, and it
will be wrong more often. Averaging them into one number would hide exactly the
decay the pipeline is built to be honest about.

The *latest* projection for a given (target bar, horizon) is the one scored: it
is the freshest estimate for that bar and the one a user would have been looking
at when it closed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from core.types import ForecastCandle

HISTORY = 500


@dataclass(slots=True)
class ProjectionScore:
    """One scored projection."""

    horizon: int
    target: pd.Timestamp
    projected_close: float
    actual_close: float
    anchor: float
    hit: bool
    error_bps: float


@dataclass(slots=True)
class HorizonStats:
    """Rolling accuracy for one projection horizon."""

    horizon: int
    scored: int = 0
    hits: int = 0
    error_bps_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.scored if self.scored else 0.0

    @property
    def mean_abs_error_bps(self) -> float:
        return self.error_bps_sum / self.scored if self.scored else 0.0

    def as_row(self) -> dict:
        return {
            "horizon": self.horizon,
            "scored": self.scored,
            "hits": self.hits,
            "misses": self.scored - self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "mean_abs_error_bps": round(self.mean_abs_error_bps, 2),
        }


@dataclass(slots=True)
class _Pending:
    """The projection for one horizon, as last recorded."""

    candle: ForecastCandle
    anchor: float


@dataclass
class ProjectionTracker:
    """Records projections and scores them as their target bars close."""

    history: int = HISTORY
    _pending: dict[pd.Timestamp, dict[int, _Pending]] = field(default_factory=dict, repr=False)
    freeze_first: bool = False
    journal: object | None = field(default=None, repr=False)
    missed_bars: int = 0
    _scores: deque[ProjectionScore] = field(default_factory=deque, repr=False)
    _stats: dict[int, HorizonStats] = field(default_factory=dict, repr=False)

    def record(self, projections: list[ForecastCandle], anchor: float, issued_at=None, context=None) -> None:
        """Remember a projection set, replacing any earlier record for the same
        (target bar, horizon) — the freshest estimate wins.
        """
        for candle in projections:
            if candle.ts is None or anchor <= 0:
                continue
            stamp = pd.Timestamp(candle.ts)
            if self.freeze_first and candle.horizon in self._pending.get(stamp, {}):
                continue
            self._pending.setdefault(stamp, {})[candle.horizon] = _Pending(candle, float(anchor))
            if self.journal is not None:
                self.journal.issued(candle, anchor, issued_at, context or {})

    def observe(self, bar_ts, close: float) -> list[ProjectionScore]:
        """Score the projections targeted at ``bar_ts``, dropping the rest.

        Called once per closed bar. Also prunes any pending record whose target
        is already in the past, so a gap in the feed cannot leave stale entries
        waiting to be scored against the wrong bar.
        """
        stamp = pd.Timestamp(bar_ts)
        pending = self._pending.pop(stamp, {})
        stale = [key for key in self._pending if key <= stamp]
        for key in stale:
            dropped = self._pending.pop(key, {})
            self.missed_bars += len(dropped)
            if self.journal is not None:
                for horizon in dropped:
                    self.journal.missing(key, horizon)

        scored: list[ProjectionScore] = []
        for horizon, record in sorted(pending.items()):
            scored.append(self._score(record, horizon, stamp, close))
        return scored

    def _score(self, record: _Pending, horizon: int, stamp: pd.Timestamp, close: float) -> ProjectionScore:
        projected = float(record.candle.close)
        anchor = record.anchor
        realised = float(close) - anchor
        lean = projected - anchor
        # A flat forecast is right only when the realised move is flat too.
        deadband = anchor * 0.00001
        def direction(value):
            return 1 if value > deadband else -1 if value < -deadband else 0
        hit = direction(realised) == direction(lean)
        error_bps = abs(close - projected) / anchor * 10_000 if anchor else 0.0

        score = ProjectionScore(
            horizon=horizon,
            target=stamp,
            projected_close=projected,
            actual_close=float(close),
            anchor=anchor,
            hit=bool(hit),
            error_bps=float(error_bps),
        )
        self._scores.append(score)
        while len(self._scores) > self.history:
            self._scores.popleft()

        stat = self._stats.setdefault(horizon, HorizonStats(horizon=horizon))
        stat.scored += 1
        stat.hits += int(hit)
        stat.error_bps_sum += error_bps
        if self.journal is not None:
            self.journal.scored(score)
        return score

    @property
    def pending(self) -> int:
        return sum(len(entries) for entries in self._pending.values())

    def stats(self) -> dict[int, HorizonStats]:
        return dict(self._stats)

    def stat(self, horizon: int) -> HorizonStats | None:
        return self._stats.get(horizon)

    def recent(self, count: int = 20) -> list[ProjectionScore]:
        return list(self._scores)[-count:]

    def overall(self) -> HorizonStats:
        total = HorizonStats(horizon=0)
        for stat in self._stats.values():
            total.scored += stat.scored
            total.hits += stat.hits
            total.error_bps_sum += stat.error_bps_sum
        return total

    def summary(self) -> str:
        overall = self.overall()
        if not overall.scored:
            return "no projections scored yet"
        return (
            f"{overall.hit_rate:.0%} directional hit over {overall.scored} scored, "
            f"{overall.mean_abs_error_bps:.1f}bp mean error"
        )

    def reset(self) -> None:
        self._pending.clear()
        self._scores.clear()
        self._stats.clear()
        self.missed_bars = 0


__all__ = ["HorizonStats", "ProjectionScore", "ProjectionTracker"]
