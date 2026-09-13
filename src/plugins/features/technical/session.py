"""Session, calendar, and regime context features.

Intraday index behaviour is strongly time-dependent: the first and last half hour
carry most of the day's range, and midday is largely noise. A model without clock
features will happily apply an opening-range pattern to a 13:00 bar. These
features give it the context to avoid that.

Weekly expiry also matters. NSE moved NIFTY weekly expiry from Thursday to
Tuesday, so the weekday is configurable rather than hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.calendar import IST, SESSION_CLOSE, SESSION_OPEN

SESSION_LENGTH_MIN = (
    (SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute)
    - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
)


def session_features(frame: pd.DataFrame, expiry_weekday: int = 1) -> pd.DataFrame:
    """Time-of-day, day-of-week, and expiry-cycle features.

    ``expiry_weekday`` uses Python's convention (0=Monday). Defaults to Tuesday,
    where NIFTY weekly expiry currently sits.
    """
    index = frame.index.tz_convert(IST)
    minutes_elapsed = (index.hour * 60 + index.minute) - (
        SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    )
    minutes_elapsed = np.clip(minutes_elapsed, 0, SESSION_LENGTH_MIN)

    progress = minutes_elapsed / SESSION_LENGTH_MIN
    weekday = index.dayofweek

    out = pd.DataFrame(index=frame.index)
    out["session_progress"] = progress
    out["minutes_from_open"] = minutes_elapsed
    out["minutes_to_close"] = SESSION_LENGTH_MIN - minutes_elapsed
    out["is_opening_30m"] = (minutes_elapsed <= 30).astype(float)
    out["is_closing_30m"] = (minutes_elapsed >= SESSION_LENGTH_MIN - 30).astype(float)
    out["is_midday_lull"] = (
        (minutes_elapsed > 60) & (minutes_elapsed < SESSION_LENGTH_MIN - 60)
    ).astype(float)

    # Cyclical encodings so 09:20 and 15:25 are not treated as far apart.
    out["time_sin"] = np.sin(2 * np.pi * progress)
    out["time_cos"] = np.cos(2 * np.pi * progress)

    out["day_of_week"] = weekday.astype(float)
    out["is_monday"] = (weekday == 0).astype(float)
    out["is_friday"] = (weekday == 4).astype(float)

    days_to_expiry = (weekday - expiry_weekday) % 7
    out["days_to_expiry"] = days_to_expiry.astype(float)
    out["is_expiry_day"] = (days_to_expiry == 0).astype(float)
    out["is_expiry_eve"] = (days_to_expiry == 6).astype(float)

    # Bars since the previous session's close, capturing overnight gaps.
    day = pd.Series(index.normalize(), index=frame.index)
    session_bar = day.ne(day.shift(1)).cumsum()
    out["bars_into_session"] = day.groupby(session_bar).cumcount().astype(float)

    return out


def overnight_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Gap and prior-session features, computed without lookahead.

    Two details matter here. Intraday high/low references use a **running**
    cumulative extreme within the session, not the whole day's extreme — using
    the full-day max would tell a 09:30 bar where the day's high ends up, which
    is exactly the kind of leak that makes a backtest look brilliant and trade
    terribly. Prior-session levels are shifted by one day for the same reason.
    """
    index = frame.index.tz_convert(IST)
    day = pd.Series(index.normalize(), index=frame.index)

    day_open = frame["open"].groupby(day).first()
    day_close = frame["close"].groupby(day).last()
    day_high = frame["high"].groupby(day).max()
    day_low = frame["low"].groupby(day).min()

    # ``day.map`` broadcasts day-level values onto every intraday bar.
    prior_close = day.map(day_close.shift(1))
    prior_open = day.map(day_open.shift(1))
    prior_high = day.map(day_high.shift(1))
    prior_low = day.map(day_low.shift(1))
    prior_prior_close = day.map(day_close.shift(2))

    running_high = frame["high"].groupby(day).cummax()
    running_low = frame["low"].groupby(day).cummin()

    out = pd.DataFrame(index=frame.index)
    out["overnight_gap"] = frame["open"] / prior_close.replace(0, np.nan) - 1.0
    out["dist_from_day_high"] = frame["close"] / running_high.replace(0, np.nan) - 1.0
    out["dist_from_day_low"] = frame["close"] / running_low.replace(0, np.nan) - 1.0
    out["day_range_position"] = (frame["close"] - running_low) / (
        (running_high - running_low).replace(0, np.nan)
    )
    out["dist_from_prior_high"] = frame["close"] / prior_high.replace(0, np.nan) - 1.0
    out["dist_from_prior_low"] = frame["close"] / prior_low.replace(0, np.nan) - 1.0
    out["prior_day_range"] = (prior_high - prior_low) / prior_close.replace(0, np.nan)
    out["prior_day_return"] = prior_close / prior_prior_close.replace(0, np.nan) - 1.0
    out["prior_day_body"] = (prior_close - prior_open) / prior_open.replace(0, np.nan)
    return out


def classify_regime(frame: pd.DataFrame, atr_col: str = "atr_norm", window: int = 100) -> pd.Series:
    """Label each bar trending / choppy / quiet using ADX and volatility rank.

    Labels are assigned causally: the volatility percentile uses only trailing data.
    """
    if "adx_14" not in frame.columns or atr_col not in frame.columns:
        return pd.Series("unknown", index=frame.index, dtype="object")

    adx_rank = frame["adx_14"].rolling(window, min_periods=20).rank(pct=True)
    vol_rank = frame[atr_col].rolling(window, min_periods=20).rank(pct=True)

    labels = np.where(
        vol_rank > 0.75,
        "volatile",
        np.where(adx_rank > 0.65, "trending", np.where(adx_rank < 0.35, "choppy", "normal")),
    )
    return pd.Series(labels, index=frame.index, dtype="object")
