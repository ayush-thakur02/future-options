"""Tick-level drift, expressed in the same units as a strategy's conviction.

A 14-bar ATR says nothing about the last twenty seconds, so a projection built
only from bar-close state would sit still while the tape moved. This tracks a
fast and a slow exponential average of tick prices and reports the distance
between them, saturated into ``[-1, 1]`` so it can be blended with a slower view.

The saturation is not cosmetic. Without it one large print would swing the
projected path by a full conviction's worth and the blue candles would flicker
rather than lean. The scale is ATR, passed in at read time rather than stored, so
the signal tightens as the market quiets down instead of staying calibrated to
whatever volatility happened to be current when it was built.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

from core.calendar import IST
from core.types import Tick

# A fast leg against a slow one: the fast leg is what moves on a burst, the slow
# leg is what the distance is measured against.
FAST_HALF_LIFE_S = 5.0
SLOW_HALF_LIFE_S = 30.0
# Distance from the slow average, in ATR units, that saturates the signal.
SATURATION_ATR = 0.35
# Fallback scale, as a fraction of price, when no ATR is supplied.
FALLBACK_SCALE_FRACTION = 1e-4
# Ticks older than this are irrelevant to a seconds-scale signal.
WINDOW = timedelta(seconds=90)


class MicroMomentum:
    """Live tick momentum, in ``[-1, 1]`` once scaled by an ATR."""

    def __init__(
        self,
        fast_half_life: float = FAST_HALF_LIFE_S,
        slow_half_life: float = SLOW_HALF_LIFE_S,
    ) -> None:
        self.fast_half_life = float(fast_half_life)
        self.slow_half_life = float(slow_half_life)
        self._fast: float | None = None
        self._slow: float | None = None
        self._last_ts: datetime | None = None
        self._recent: deque[tuple[datetime, float]] = deque()
        self.ticks = 0

    def update(self, tick: Tick) -> None:
        """Fold in one tick."""
        price = tick.ltp or tick.mid
        if not price:
            return

        moment = tick.ts.astimezone(IST) if tick.ts.tzinfo else tick.ts.replace(tzinfo=IST)
        if self._last_ts is None:
            self._fast = self._slow = float(price)
        else:
            elapsed = max((moment - self._last_ts).total_seconds(), 1e-3)
            self._fast = _decay(self._fast, price, elapsed, self.fast_half_life)
            self._slow = _decay(self._slow, price, elapsed, self.slow_half_life)

        self._last_ts = moment
        self._recent.append((moment, float(price)))
        while self._recent and moment - self._recent[0][0] > WINDOW:
            self._recent.popleft()
        self.ticks += 1

    @property
    def distance(self) -> float:
        """Fast minus slow, in price units."""
        if self._fast is None or self._slow is None:
            return 0.0
        return self._fast - self._slow

    @property
    def spread_bps(self) -> float:
        """Fast-versus-slow distance in basis points, for display."""
        if not self._slow:
            return 0.0
        return (self._fast / self._slow - 1.0) * 10_000

    def value(self, atr: float = 0.0) -> float:
        """Signed momentum in ``[-1, 1]``, scaled by ``atr``."""
        scale = atr if atr > 0 else (self._slow or 0.0) * FALLBACK_SCALE_FRACTION * 10.0
        if scale <= 0:
            return 0.0
        return float(math.tanh(self.distance / (SATURATION_ATR * scale)))

    def reset(self) -> None:
        self._fast = self._slow = None
        self._last_ts = None
        self._recent.clear()
        self.ticks = 0

    def __repr__(self) -> str:
        return f"<MicroMomentum {self.spread_bps:+.2f}bp over {self.ticks} ticks>"


def _decay(current: float, target: float, elapsed: float, half_life: float) -> float:
    """Exponential average that is correct for irregular tick spacing.

    Weighting every tick equally would let a burst of prints at one instant look
    like a trend that lasted a minute.
    """
    if half_life <= 0:
        return float(target)
    weight = 1.0 - 0.5 ** (elapsed / half_life)
    return current + (float(target) - current) * weight


__all__ = ["MicroMomentum"]
