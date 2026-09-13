"""Projecting the next few candles.

A projection is not a prediction of a price path. It is a *path estimate*: given
where price is now, how strong the current view is, and how much the market is
moving, where are the next few bars likely to travel?

Three properties make it useful rather than decorative:

* **Anchored on the live price.** The first projected bar opens where the tape
  is right now, so the projection moves with the market instead of drifting away
  from it between bar closes.
* **Damped with horizon.** Conviction for the bar after next is worth less than
  conviction for the next one, and the third bar less again. A projection that
  keeps full conviction three bars out is claiming to know more than it does.
* **Widened with horizon.** Uncertainty compounds, so the projected range grows
  with distance. Far bars are drawn taller because they are genuinely less
  certain, not to look impressive.

Everything here is deterministic given its inputs. That matters: a projection
that jitters randomly each second would be impossible to evaluate, and the whole
point of :mod:`~plugins.forecasts.projection.tracker` is that this one can be
scored against what actually happened.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from core.calendar import IST
from core.types import Direction, ForecastCandle
from plugins.aggregators.candle_builder import bar_start

# How much of a full-conviction view is realised in one bar, as a fraction of ATR.
DRIFT_SCALE = 0.62
# Conviction surviving each additional bar of horizon.
HORIZON_DECAY = 0.72
# Range of the nearest projected bar, as a multiple of ATR.
BASE_RANGE = 0.85
# Extra range added per bar of horizon, as a multiple of ATR.
RANGE_GROWTH = 0.30
# Fraction of the range extending beyond the body on each side.
WICK_FRACTION = 0.32
# How much live tick momentum overrides the slower conviction view.
MICRO_WEIGHT = 0.35
# Volatility is held in this band relative to its own recent median.
VOL_SCALE_BOUNDS = (0.55, 2.20)

DEFAULT_BARS_AHEAD = 3


def atr_of(bars: pd.DataFrame, window: int = 14) -> float:
    """Average true range per bar, in price units."""
    if bars.empty:
        return 0.0
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    close = bars["close"].astype("float64")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    value = float(true_range.tail(window).mean())
    if not np.isfinite(value) or value <= 0.0:
        # A flat or one-bar series has no range to measure; fall back to a small
        # fraction of price so the projection collapses to a flat line rather
        # than exploding or vanishing.
        price = float(close.iloc[-1]) if len(close) else 0.0
        return price * 1e-5
    return value


def volatility_scale(bars: pd.DataFrame, atr: float, window: int = 50) -> float:
    """Current ATR against its own recent median, clamped.

    Elevated volatility should widen a projection, not leave it looking as calm
    as a quiet afternoon. The clamp stops a single violent bar from producing an
    absurd range.
    """
    if bars.empty or atr <= 0:
        return 1.0
    true_range = (bars["high"] - bars["low"]).astype("float64").abs()
    recent = float(true_range.tail(window).median())
    if not np.isfinite(recent) or recent <= 0:
        return 1.0
    value = float(np.clip(atr / recent, *VOL_SCALE_BOUNDS))
    return value if np.isfinite(value) else 1.0


def project_candles(
    bars: pd.DataFrame,
    anchor: float,
    conviction: float,
    now=None,
    bars_ahead: int = DEFAULT_BARS_AHEAD,
    bar_minutes: int = 1,
) -> list[ForecastCandle]:
    """Project the next ``bars_ahead`` candles from the current state.

    ``bars`` is used only for volatility: the path itself starts at ``anchor``,
    which is the live price. That is what keeps a projected candle moving with
    the tape rather than with the last completed bar.
    """
    if bars_ahead <= 0 or anchor <= 0:
        return []

    atr = atr_of(bars)
    scale = volatility_scale(bars, atr)
    view = float(np.clip(conviction, -1.0, 1.0))
    moment = (now or datetime.now(IST)).astimezone(IST)
    start = bar_start(moment, bar_minutes)

    projected: list[ForecastCandle] = []
    previous_close = float(anchor)

    for step in range(1, bars_ahead + 1):
        decay = HORIZON_DECAY ** (step - 1)
        drift = view * decay * DRIFT_SCALE * atr

        open_price = previous_close
        close_price = open_price + drift

        span = atr * (BASE_RANGE + RANGE_GROWTH * (step - 1)) * scale
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        high = body_top + span * WICK_FRACTION
        low = max(body_bottom - span * WICK_FRACTION, 0.0)

        confidence = float(np.clip(abs(view) * decay, 0.0, 1.0))
        move_bps = (close_price / open_price - 1.0) * 10_000 if open_price else 0.0

        projected.append(
            ForecastCandle(
                ts=start + pd.Timedelta(minutes=bar_minutes * step),
                horizon=step,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                conviction=view * decay,
                confidence=confidence,
                expected_move_bps=float(move_bps),
                bar_minutes=bar_minutes,
            )
        )
        previous_close = close_price

    return projected


def projected_frame(projections: list[ForecastCandle], dtype="float64") -> pd.DataFrame:
    """The projected candles as an OHLCV frame, for the chart layer."""
    if not projections:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"], dtype=dtype)

    index = pd.DatetimeIndex([candle.ts for candle in projections], tz=IST, name="ts")
    rows = [candle.as_row() for candle in projections]
    return pd.DataFrame(rows, index=index, dtype=dtype)


def projection_summary(projections: list[ForecastCandle]) -> str:
    """A one-line description of where the path ends up."""
    if not projections:
        return "no projection"
    last = projections[-1]
    direction = Direction.from_value(last.close - projections[0].open)
    total_bps = (last.close / projections[0].open - 1.0) * 10_000
    return f"{direction.value} {total_bps:+.1f}bp over {len(projections)} bars"


__all__ = [
    "BASE_RANGE",
    "DEFAULT_BARS_AHEAD",
    "DRIFT_SCALE",
    "HORIZON_DECAY",
    "MICRO_WEIGHT",
    "RANGE_GROWTH",
    "atr_of",
    "project_candles",
    "projected_frame",
    "projection_summary",
    "volatility_scale",
]
