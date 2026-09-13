"""Where the projected path gets its direction.

Three inputs at three cadences, and each earns its place:

* the **ensemble** — every strategy blended, plus the model when one is trained.
  The considered view, and only recomputed when a bar closes, because that is
  when there is new information to judge.
* the **bar trend** — fast and slow EMAs over the recent bars, read on every
  refresh. A one-minute bar is a long time to hold a stale opinion, and this is
  what lets the projected path lean within the bar rather than only at the end.
* **tick momentum** — inside the projection pack, which is why this module stops
  at the bar scale.

The blend is deliberately simple and monotone. A more elaborate weighting would
be untestable against a target that is itself noisy, and the property that
matters is only that the projected path moves when the market moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Weight given to the bar-scale trend against the ensemble's considered view.
FAST_WEIGHT = 0.35
# Distance between the EMAs, in ATR units, that saturates the trend read.
SATURATION_ATR = 1.20
MIN_BARS = 12


def trend_conviction(
    bars: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
    window: int = 60,
) -> float:
    """Signed bar-scale trend in ``[-1, 1]``.

    ``fast`` against ``slow`` EMA distance, scaled by ATR so the reading does not
    change meaning when volatility does. Falls back to a short-window price
    comparison when there are too few bars for the EMAs, which is the case for
    the first seconds of a replay.
    """
    if bars is None or len(bars) < 2:
        return 0.0

    close = bars["close"].astype("float64")
    if len(close) >= max(slow, MIN_BARS):
        fast_ema = close.ewm(span=fast, adjust=False).mean().iloc[-1]
        slow_ema = close.ewm(span=slow, adjust=False).mean().iloc[-1]
        distance = float(fast_ema - slow_ema)
    else:
        distance = float(close.iloc[-1] - close.iloc[0])

    atr = _atr(bars, window)
    if atr <= 0:
        return 0.0
    return float(np.tanh(distance / (SATURATION_ATR * atr)))


def blend(slow: float, fast: float, fast_weight: float = FAST_WEIGHT) -> float:
    """Combine the considered view with the live bar trend."""
    weight = float(np.clip(fast_weight, 0.0, 1.0))
    combined = (1.0 - weight) * float(np.clip(slow, -1.0, 1.0)) + weight * float(
        np.clip(fast, -1.0, 1.0)
    )
    return float(np.clip(combined, -1.0, 1.0))


def _atr(bars: pd.DataFrame, window: int = 60, default_fraction: float = 1e-4) -> float:
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    previous = bars["close"].astype("float64").shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    value = float(true_range.tail(window).mean())
    if not np.isfinite(value) or value <= 0:
        price = float(bars["close"].iloc[-1]) if len(bars) else 0.0
        return price * default_fraction
    return value


__all__ = ["FAST_WEIGHT", "SATURATION_ATR", "blend", "trend_conviction"]
