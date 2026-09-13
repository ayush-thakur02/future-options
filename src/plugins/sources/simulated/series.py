"""Synthetic market data source for offline development.

Generates NIFTY-like 1-minute bars with volatility clustering, intraday volume
seasonality, and regime shifts. The point is not realism for its own sake — it is
to have a series with genuine (if weak) autocorrelation structure so the full
pipeline, including model training and backtesting, can be exercised end to end
without live credentials.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST, SESSION_OPEN

BASE_LEVEL = 24_000.0


def _session_days(count: int, end: date | None = None) -> list[date]:
    end = end or datetime.now(IST).date()
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def _intraday_volume_shape(bars_per_day: int) -> np.ndarray:
    """U-shaped activity curve: busy at the open and close, quiet midday."""
    x = np.linspace(0.0, 1.0, bars_per_day)
    return 0.45 + 0.9 * ((x - 0.5) ** 2) * 4.0


def rebase(frame: pd.DataFrame, price: float) -> pd.DataFrame:
    """Scale a series so it opens at ``price``.

    Used to continue a replay where real history left off. A *ratio* rather than
    an offset: the series is multiplicative, and shifting it by a constant would
    change the size of every move in it, which is the one thing the generated
    shape is for.
    """
    if frame.empty or price <= 0:
        return frame
    first = float(frame["open"].iloc[0])
    if first <= 0:
        return frame
    factor = float(price) / first
    if not 0.1 < factor < 10.0:  # a wild factor means the wrong frame was passed
        return frame
    scaled = frame.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * factor
    return scaled


def generate_candles(
    days: int = 120,
    bar_minutes: int = 1,
    seed: int = 7,
    end: date | None = None,
    with_volume: bool = False,
) -> pd.DataFrame:
    """Build a synthetic 1-minute OHLCV series spanning ``days`` sessions."""
    rng = np.random.default_rng(seed)
    bars_per_day = (6 * 60 + 15) // bar_minutes  # 09:15 -> 15:30
    sessions = _session_days(days, end=end)
    total = len(sessions) * bars_per_day

    # Regime-switching drift and volatility, matching how index vol actually
    # clusters rather than staying constant.
    n_regimes = 5
    regime_len = max(total // n_regimes, 1)
    drifts = rng.normal(0.0, 1.4, n_regimes)
    vols = np.abs(rng.normal(1.0, 0.35, n_regimes)) + 0.35
    regime = np.repeat(np.arange(n_regimes), regime_len)[:total]

    drift = drifts[regime] * 1e-6 * (bar_minutes**0.5)
    vol = vols[regime] * 1.1e-4 * (bar_minutes**0.5)

    # Volatility clustering via a simple AR(1) process on the shock scale.
    shocks = rng.standard_normal(total)
    scale = np.ones(total)
    for i in range(1, total):
        scale[i] = 0.93 * scale[i - 1] + 0.07 * (1.0 + 0.5 * abs(shocks[i - 1]))
    scale = np.clip(scale, 0.4, 3.5)

    returns = drift + vol * scale * shocks

    # A weak mean-reverting component gives strategies something to find.
    returns[1:] -= 0.015 * returns[:-1]

    closes = BASE_LEVEL * np.exp(np.cumsum(returns))

    volume_shape = np.tile(_intraday_volume_shape(bars_per_day), len(sessions))
    volume_noise = rng.lognormal(0.0, 0.45, total)
    volume = (volume_shape * volume_noise * 45_000).round()

    timestamps: list[datetime] = []
    for day in sessions:
        open_dt = datetime.combine(day, SESSION_OPEN, tzinfo=IST)
        timestamps.extend([open_dt + timedelta(minutes=i * bar_minutes) for i in range(bars_per_day)])

    frame = pd.DataFrame(index=pd.DatetimeIndex(timestamps, name="ts"))
    frame["close"] = closes

    opens = np.empty(total)
    opens[0] = BASE_LEVEL
    opens[1:] = closes[:-1]
    frame["open"] = opens

    intrabar = np.abs(rng.normal(0.0, 0.55, total)) * np.abs(returns) + 1e-9
    frame["high"] = np.maximum(frame["open"], frame["close"]) + frame["close"] * intrabar
    frame["low"] = np.minimum(frame["open"], frame["close"]) - frame["close"] * intrabar
    frame["volume"] = volume if with_volume else 0.0
    frame["oi"] = 0.0

    frame = frame[["open", "high", "low", "close", "volume", "oi"]]
    return normalize_candles(frame)


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
