"""Turn the locally recorded WebSocket tape into reusable candle history.

The tape is the highest-fidelity local source: once a tick has been received, a
later startup should not ask the broker for a candle covering the same minute.
This module owns that bridge while remaining provider-independent. It knows the
``TickRecorder`` schema and the canonical candle schema, but nothing about
Upstox or any other broker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.bars import empty_frame, normalize_candles
from core.calendar import IST, SESSION_CLOSE, SESSION_OPEN

from .partitions import HOUR, TICKS, PartitionedStore, index_stamps, normalize_frames
from .recorder import recover_tick_journal


class RealtimeArchive:
    """Recorded ticks for one instrument, materialized into closed candles."""

    def __init__(self, root: Path, instrument: str) -> None:
        self.store = PartitionedStore(
            root,
            instrument,
            dataset=TICKS,
            shard=HOUR,
            normalizer=normalize_frames,
        )

    def recover(self) -> int:
        """Recover complete records left in an interrupted recorder journal."""
        return recover_tick_journal(self.store)

    def candles(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        bar_minutes: int = 1,
    ) -> pd.DataFrame:
        return ticks_to_candles(self.store.load(start=start, end=end), bar_minutes=bar_minutes)

    def materialize(
        self,
        candle_store: PartitionedStore,
        now: pd.Timestamp | None = None,
    ) -> int:
        """Merge every newly completed tape bar into ``candle_store``.

        The current minute stays on the tape until it closes. This prevents a
        restart at 09:15:20 from warming the engine with a partial 09:15 candle
        and then rejecting the remaining ticks in that minute as historical.
        """
        self.recover()
        first_tick, last_tick = self.store.coverage()
        if first_tick is None or last_tick is None:
            return 0

        now = _as_ist(now or pd.Timestamp.now(tz=IST))
        interval = pd.Timedelta(minutes=max(int(candle_store.bar_minutes), 1))
        current_bar = now.floor(f"{int(interval.total_seconds() // 60)}min")
        eligible_last = min(last_tick.floor(f"{int(interval.total_seconds() // 60)}min"), current_bar - interval)
        if eligible_last < first_tick.floor("min"):
            return 0

        candle_entry = candle_store.manifest.dataset(candle_store.dataset, candle_store.key)
        checkpoint = candle_entry.get("realtime_archive") or {}
        tape_rows = self.store.row_count()
        checkpoint_last = checkpoint.get("last_bar")
        if (
            int(checkpoint.get("tick_rows", -1)) == tape_rows
            and checkpoint_last
            and _as_ist(checkpoint_last) >= eligible_last
        ):
            return 0

        start = first_tick
        if checkpoint_last:
            # Rebuild the checkpoint bar as late ticks may have updated it before
            # the hour shard was finally flushed.
            start = max(first_tick, _as_ist(checkpoint_last))
        ticks = self.store.load(start=start, end=eligible_last + interval - pd.Timedelta(nanoseconds=1))
        bars = ticks_to_candles(ticks, bar_minutes=candle_store.bar_minutes)
        bars = bars[bars.index <= eligible_last]
        if bars.empty:
            return 0

        candle_store.write(bars)
        previous_rows = int(checkpoint.get("bars_materialized", 0) or 0)
        candle_store.manifest.update(
            candle_store.dataset,
            candle_store.key,
            realtime_archive={
                "dataset": TICKS,
                "first_tick": first_tick.isoformat(),
                "last_tick": last_tick.isoformat(),
                "last_bar": bars.index[-1].isoformat(),
                "tick_rows": tape_rows,
                "bars_materialized": previous_rows + len(bars),
            },
        )
        return len(bars)


def ticks_to_candles(ticks: pd.DataFrame, bar_minutes: int = 1) -> pd.DataFrame:
    """Aggregate exchange updates into session-anchored OHLCV candles."""
    if ticks is None or ticks.empty or "ltp" not in ticks.columns:
        return empty_frame()
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be at least one")

    frame = ticks.copy()
    frame.index = index_stamps(frame.index)
    frame.index = _as_ist_index(frame.index)
    frame = frame.sort_index(kind="stable")
    frame["ltp"] = pd.to_numeric(frame["ltp"], errors="coerce")
    frame = frame[frame["ltp"].notna() & (frame["ltp"] > 0)]
    if frame.empty:
        return empty_frame()

    session_mask = (frame.index.time >= SESSION_OPEN) & (frame.index.time < SESSION_CLOSE)
    frame = frame.loc[session_mask]
    if frame.empty:
        return empty_frame()

    opened = frame.index.normalize() + pd.Timedelta(hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute)
    elapsed = ((frame.index - opened).total_seconds() // 60).astype("int64")
    keys = opened + pd.to_timedelta((elapsed // bar_minutes) * bar_minutes, unit="min")
    keys.name = "ts"
    grouped = frame.groupby(keys, sort=True)
    candles = pd.DataFrame(
        {
            "open": grouped["ltp"].first(),
            "high": grouped["ltp"].max(),
            "low": grouped["ltp"].min(),
            "close": grouped["ltp"].last(),
        }
    )

    if "volume" in frame.columns:
        cumulative = pd.to_numeric(grouped["volume"].last(), errors="coerce").fillna(0.0)
        # WebSocket volume is cumulative for the session. The first archived
        # minute may be mid-session, so its unknown pre-archive volume is zero.
        volume = cumulative.groupby(cumulative.index.date).diff().fillna(0.0)
        candles["volume"] = volume.clip(lower=0.0)
    else:
        candles["volume"] = 0.0
    candles["oi"] = (
        pd.to_numeric(grouped["oi"].last(), errors="coerce").fillna(0.0)
        if "oi" in frame.columns
        else 0.0
    )
    return normalize_candles(candles)


def _as_ist(moment) -> pd.Timestamp:
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)


def _as_ist_index(index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(index, name="ts")
    return stamps.tz_localize(IST) if stamps.tz is None else stamps.tz_convert(IST)


__all__ = ["RealtimeArchive", "ticks_to_candles"]
