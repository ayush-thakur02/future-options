"""Tick-to-candle aggregation.

One subtlety drives this design: the NIFTY 50 index has no traded volume. There
is no consolidated tape for an index, so ``volume`` arrives as zero. Rather than
silently feed zero-volume bars into volume-based indicators (which would produce
a constant, meaningless signal), we also record ``tick_count`` — the number of
updates that landed in the bar. That is a genuine proxy for activity and is what
the feature layer falls back to when volume is unavailable.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

import pandas as pd

from core.calendar import IST, SESSION_CLOSE, SESSION_OPEN
from core.types import Tick


def bar_start(moment: datetime, bar_minutes: int) -> datetime:
    """Timestamp of the bar that ``moment`` belongs to, anchored to session open."""
    moment = moment.astimezone(IST)
    session_open = moment.replace(
        hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0, microsecond=0
    )
    elapsed = (moment - session_open).total_seconds()
    bucket = int(elapsed // (bar_minutes * 60))
    return session_open + pd.Timedelta(minutes=bucket * bar_minutes)


class CandleAggregator:
    """Accumulates ticks into OHLCV bars and emits each bar once it closes."""

    def __init__(self, bar_minutes: int = 1, max_bars: int = 2000) -> None:
        self.bar_minutes = bar_minutes
        self._bars: deque[dict] = deque(maxlen=max_bars)
        self._current: dict | None = None
        self.tick_count = 0
        self.last_tick: Tick | None = None
        self._cumulative_volume: float | None = None
        self._volume_day = None
        self._volume_delta = 0.0

    @property
    def current_bar(self) -> dict | None:
        return self._current

    def on_tick(self, tick: Tick) -> dict | None:
        """Feed a tick. Returns the previous bar when a new one begins."""
        if self.last_tick is not None and tick.ts < self.last_tick.ts:
            return None
        day = tick.ts.astimezone(IST).date()
        if self._volume_day != day:
            self._cumulative_volume = None
            self._volume_day = day
        current_volume = float(tick.volume_traded)
        self._volume_delta = max(current_volume - self._cumulative_volume, 0) if self._cumulative_volume is not None and current_volume > 0 else 0.0
        if current_volume > 0:
            self._cumulative_volume = current_volume
        self.tick_count += 1
        self.last_tick = tick

        start = bar_start(tick.ts, self.bar_minutes)
        closed: dict | None = None

        if self._current is None:
            self._current = self._new_bar(start, tick)
        elif start > self._current["ts"]:
            # Session gap: drop a stale partial bar rather than emit a broken one.
            if self._is_same_session(self._current["ts"], start):
                closed = self._finalize(self._current)
                self._bars.append(closed)
            self._current = self._new_bar(start, tick)
        else:
            self._update(self._current, tick)

        return closed

    def _new_bar(self, start: datetime, tick: Tick) -> dict:
        price = tick.ltp or tick.mid
        return {
            "ts": start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": self._volume_delta,
            "oi": float(tick.open_interest),
            "tick_count": 1,
            "buy_qty": float(tick.total_buy_qty),
            "sell_qty": float(tick.total_sell_qty),
            "bid_qty": float(tick.bid_q),
            "ask_qty": float(tick.ask_q),
        }

    def _update(self, bar: dict, tick: Tick) -> None:
        price = tick.ltp or tick.mid
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["tick_count"] += 1
        # vtt is daily cumulative; historical candle volume is per bar.
        # Keeping the daily total here corrupts VWAP and train/serve parity.
        bar["volume"] += self._volume_delta
        if tick.open_interest:
            bar["oi"] = float(tick.open_interest)
        bar["buy_qty"] = float(tick.total_buy_qty)
        bar["sell_qty"] = float(tick.total_sell_qty)
        bar["bid_qty"] = float(tick.bid_q)
        bar["ask_qty"] = float(tick.ask_q)

    def _finalize(self, bar: dict) -> dict:
        return dict(bar)

    def _is_same_session(self, previous: datetime, current: datetime) -> bool:
        return previous.astimezone(IST).date() == current.astimezone(IST).date()

    def is_stale(self, now: datetime | None = None) -> bool:
        """True when no tick has arrived for more than two bar intervals."""
        if self._current is None:
            return True
        now = now or datetime.now(IST)
        gap = (now - self._current["ts"]).total_seconds()
        return gap > self.bar_minutes * 60 * 2

    def closed_bars(self) -> pd.DataFrame:
        if not self._bars:
            return _empty_bars()
        frame = pd.DataFrame(list(self._bars)).set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert(IST)
        return frame.sort_index()

    def snapshot_frame(self, include_current: bool = True) -> pd.DataFrame:
        """Closed bars plus, optionally, the forming bar."""
        rows = list(self._bars)
        if include_current and self._current is not None:
            rows = rows + [self._current]
        if not rows:
            return _empty_bars()
        frame = pd.DataFrame(rows).set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert(IST)
        return frame.sort_index()

    def bar_expected_to_close(self, now: datetime | None = None) -> bool:
        if self._current is None:
            return False
        now = now or datetime.now(IST)
        if now.astimezone(IST).time() > SESSION_CLOSE:
            return True
        return bar_start(now, self.bar_minutes) > self._current["ts"]


def _empty_bars() -> pd.DataFrame:
    columns = [
        "open", "high", "low", "close", "volume", "oi",
        "tick_count", "buy_qty", "sell_qty", "bid_qty", "ask_qty",
    ]
    frame = pd.DataFrame(columns=columns, dtype="float64")
    frame.index = pd.DatetimeIndex([], tz=IST, name="ts")
    return frame
