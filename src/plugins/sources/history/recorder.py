"""Writing the live tape and the chain as they arrive.

Candles can be re-fetched. **Ticks cannot.** A provider publishes historical
candles; it does not publish the tape that produced them, so whatever is not
recorded while it is streaming is gone for good. That asymmetry is the reason
recording is not optional on a live run.

Both recorders buffer and flush rather than writing per event. A parquet write
costs milliseconds and a NIFTY tick arrives every few milliseconds at the open, so
a write per tick would spend the whole session in the filesystem. The buffer is
bounded by rows *and* by the hour boundary — the boundary matters because it is
also the partition key, so a late flush would have to write into a closed shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.calendar import IST
from core.types import Tick

from .partitions import CHAIN, HOUR, TICKS, PartitionedStore, normalize_frames

# Rows held in memory before a flush. A minute of a busy index feed is a few
# thousand ticks, so this bounds the exposure to a few seconds of data.
DEFAULT_BUFFER = 4_000
# How often a chain snapshot is taken, in seconds. A chain is not a tick stream:
# open interest and implied vol move on a scale of minutes, and polling faster
# would mostly record the same numbers again.
CHAIN_SAMPLE_SECONDS = 60.0


@dataclass
class TickRecorder:
    """Every tick that arrives, buffered and written to hour shards."""

    store: PartitionedStore
    buffer_size: int = DEFAULT_BUFFER
    rows_written: int = 0
    flushes: int = 0
    dropped: int = 0
    _buffer: list[dict] = field(default_factory=list, repr=False)
    _hour: str = field(default="", repr=False)

    def record(self, tick: Tick) -> None:
        """Add one tick, flushing when the buffer fills or the hour turns."""
        moment = tick.ts.astimezone(IST)
        hour = f"{moment:%Y-%m-%dT%H}"
        if hour != self._hour and self._buffer:
            # The hour is the partition key, so a buffered row from the previous
            # hour must land before this one starts arriving.
            self.flush()

        self._buffer.append(
            {
                "ts": moment,
                "ltp": float(tick.ltp),
                "ltq": int(tick.ltq),
                "close_prev": float(tick.close_prev),
                "bid_p": float(tick.bid_p),
                "bid_q": int(tick.bid_q),
                "ask_p": float(tick.ask_p),
                "ask_q": int(tick.ask_q),
                "volume": int(tick.volume_traded),
                "atp": float(tick.avg_traded_price),
                "oi": float(tick.open_interest),
                "buy_qty": float(tick.total_buy_qty),
                "sell_qty": float(tick.total_sell_qty),
            }
        )
        self._hour = hour
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> int:
        """Write the buffer out. Returns the rows written."""
        if not self._buffer:
            return 0
        frame = pd.DataFrame(self._buffer).set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index, tz=IST, name="ts")
        written = self.store.write(normalize_frames(frame))
        self.rows_written += len(self._buffer)
        self.flushes += 1
        self._buffer.clear()
        return written

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        self.flush()

    def describe(self) -> str:
        coverage = self.store.coverage()
        return (
            f"{self.store.instrument}: {self.rows_written:,} ticks written, "
            f"{self.flushes} flushes, {self.buffered} buffered"
            + (f", {coverage[0]:%H:%M}→{coverage[1]:%H:%M}" if coverage[0] is not None else "")
        )


@dataclass
class ChainRecorder:
    """Chain snapshots, sampled rather than streamed."""

    store: PartitionedStore
    sample_seconds: float = CHAIN_SAMPLE_SECONDS
    rows_written: int = 0
    samples: int = 0
    _last_sample: float = field(default=-1e9, repr=False)

    def due(self, now_seconds: float) -> bool:
        return (now_seconds - self._last_sample) >= self.sample_seconds

    def record(self, chain, now_seconds: float) -> int:
        """Write one snapshot of every strike. Returns the rows written."""
        self._last_sample = now_seconds
        rows = []
        for strike in chain.strikes():
            for kind in ("CE", "PE"):
                quote = chain.at(strike, kind)
                if quote is None:
                    continue
                rows.append(
                    {
                        "ts": chain.ts,
                        "expiry": chain.expiry,
                        "spot": quote.spot,
                        "strike": quote.strike,
                        "kind": quote.kind,
                        "premium": quote.premium,
                        "delta": quote.delta,
                        "gamma": quote.gamma,
                        "theta_per_minute": quote.theta_per_minute,
                        "vega": quote.vega,
                        "iv": quote.iv,
                        "oi": quote.oi,
                        "minutes_to_expiry": quote.minutes_to_expiry,
                    }
                )
        if not rows:
            return 0

        # Indexed by (ts, strike, kind): one snapshot holds many strikes at the
        # same instant, and an index of time alone would keep only the last.
        frame = pd.DataFrame(rows).set_index(["ts", "strike", "kind"])
        frame.index = frame.index.set_levels(
            frame.index.levels[0].tz_localize(IST)
            if frame.index.levels[0].tz is None
            else frame.index.levels[0].tz_convert(IST),
            level=0,
        )
        written = self.store.write(normalize_frames(frame))
        self.rows_written += len(rows)
        self.samples += 1
        return written

    def close(self) -> None:
        return None

    def describe(self) -> str:
        return f"{self.store.instrument}: {self.samples} chain samples, {self.rows_written:,} rows"


__all__ = [
    "CHAIN_SAMPLE_SECONDS",
    "DEFAULT_BUFFER",
    "CHAIN",
    "ChainRecorder",
    "HOUR",
    "TICKS",
    "TickRecorder",
]
