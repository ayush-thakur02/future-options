"""Tick generation for a synthetic series.

A bar is not a single price — a real minute contains hundreds of updates. These
functions expand a bar series into a tick path, which is what lets the replay
exercise the aggregator, the forming candle, and the projection exactly as a live
feed would.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.calendar import IST


def generate_ticks(
    bars: pd.DataFrame,
    ticks_per_bar: int = 6,
    seed: int = 11,
) -> pd.DataFrame:
    """Expand synthetic bars into a plausible tick stream for feed-replay mode."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for ts, bar in bars.iterrows():
        path = np.linspace(bar["open"], bar["close"], ticks_per_bar)
        jitter = rng.normal(0.0, abs(bar["high"] - bar["low"]) * 0.08 + 1e-6, ticks_per_bar)
        for offset, price in enumerate(path + jitter):
            rows.append(
                {
                    "ts": ts + timedelta(seconds=int(60 * offset / ticks_per_bar)),
                    "ltp": float(price),
                    "ltq": int(rng.integers(1, 40)),
                    "close_prev": float(bars["close"].iloc[0]),
                    "bid_p": float(price) - 0.05,
                    "bid_q": int(rng.integers(50, 400)),
                    "ask_p": float(price) + 0.05,
                    "ask_q": int(rng.integers(50, 400)),
                    "volume_traded": int(rng.integers(1000, 9000)),
                    "avg_traded_price": float(price),
                    "open_interest": 0.0,
                    "total_buy_qty": float(rng.integers(10_000, 90_000)),
                    "total_sell_qty": float(rng.integers(10_000, 90_000)),
                }
            )
    frame = pd.DataFrame(rows).set_index("ts")
    frame.index = pd.DatetimeIndex(frame.index).tz_convert(IST)
    return frame.sort_index()


def generate_quote(price: float, prev_close: float) -> dict:
    """A quote payload shaped like the Upstox REST response."""
    return {
        "last_price": price,
        "ohlc": {
            "open": prev_close * 1.001,
            "high": max(price, prev_close) * 1.004,
            "low": min(price, prev_close) * 0.996,
            "close": prev_close,
        },
        "volume": 0,
        "close_price": prev_close,
    }


__all__ = ["generate_quote", "generate_ticks"]
