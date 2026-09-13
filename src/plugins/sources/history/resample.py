"""OHLCV resampling that respects NSE session boundaries.

Naive ``resample("5min")`` is wrong for Indian markets. It anchors buckets to the
top of the hour rather than to the 09:15 open, so every bar is offset by 45
minutes and the last bar of the day becomes a 15-minute stub. It also happily
merges a 15:29 bar with the next morning's 09:16 bar, leaking information across
the overnight gap.

Both are avoided by computing the bar timestamp for each row directly — session
open plus the elapsed bucket — and grouping on that. Because the bucket counter
restarts at each session open, bars can never span two days, and no separate
session guard is needed.
"""

from __future__ import annotations

import pandas as pd

from core.calendar import IST, SESSION_OPEN

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "oi"]


def _as_ist_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def bar_timestamps(index: pd.DatetimeIndex, bar_minutes: int) -> pd.DatetimeIndex:
    """Bar open time for each row, anchored to the 09:15 session open."""
    session_open = index.normalize() + pd.Timedelta(
        hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute
    )
    elapsed_minutes = (index - session_open).total_seconds() / 60.0
    bucket = (elapsed_minutes // bar_minutes).astype("int64")
    return session_open + pd.to_timedelta(bucket * bar_minutes, unit="m")


def resample_ohlcv(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV rows into ``bar_minutes`` buckets.

    Returns a frame indexed by bar open time in IST — the timestamp a chart shows
    and a strategy expects to act on.
    """
    if df.empty:
        return df.copy()
    if bar_minutes <= 1:
        return df.sort_index()

    frame = df.sort_index()
    frame.index = _as_ist_index(frame.index)

    keys = bar_timestamps(frame.index, bar_minutes)
    keys.name = "ts"
    grouped = frame.groupby(keys, sort=True)

    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )

    for column, aggregator in (("volume", "sum"), ("trades", "sum"), ("tick_count", "sum"), ("oi", "last")):
        if column in frame.columns:
            out[column] = getattr(grouped[column], aggregator)()

    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "oi" not in out.columns:
        out["oi"] = 0.0

    out.index.name = "ts"
    return out.sort_index()


def align_to_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows inside regular NSE session hours."""
    if df.empty:
        return df
    idx = _as_ist_index(df.index)
    times = idx.time
    mask = (times >= SESSION_OPEN) & (times <= pd.Timestamp("15:30").time())
    return df.loc[mask]
