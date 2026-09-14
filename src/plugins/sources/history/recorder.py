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

import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.calendar import IST
from core.types import Tick

from .partitions import CHAIN, HOUR, TICKS, PartitionedStore, normalize_frames

# Rows held in memory before a flush. A minute of a busy index feed is a few
# thousand ticks, so this bounds the exposure to a few seconds of data.
DEFAULT_BUFFER = 4_000
# The journal is unbuffered and append-only. Syncing it periodically protects
# against a host crash without forcing a disk barrier for every market update;
# an ordinary process crash loses no complete journal writes.
DEFAULT_JOURNAL_SYNC = 256
# How often a chain snapshot is taken, in seconds. A chain is not a tick stream:
# open interest and implied vol move on a scale of minutes, and polling faster
# would mostly record the same numbers again.
CHAIN_SAMPLE_SECONDS = 60.0


@dataclass
class TickRecorder:
    """Every tick that arrives, buffered and written to hour shards."""

    store: PartitionedStore
    buffer_size: int = DEFAULT_BUFFER
    journal_sync_every: int = DEFAULT_JOURNAL_SYNC
    rows_written: int = 0
    rows_recovered: int = 0
    flushes: int = 0
    dropped: int = 0
    _buffer: list[dict] = field(default_factory=list, repr=False)
    _hour: str = field(default="", repr=False)
    _journal_fd: int | None = field(default=None, init=False, repr=False)
    _journal_writes: int = field(default=0, init=False, repr=False)
    _lock_fd: int | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be at least one")
        if self.journal_sync_every < 1:
            raise ValueError("journal_sync_every must be at least one")
        self._lock_fd = _lock_journal(self.store, blocking=False)
        if self._lock_fd is None:
            raise RuntimeError(f"another tick recorder already owns {self.store.instrument}")
        self.rows_recovered = _recover_tick_journal_locked(self.store)

    def record(self, tick: Tick) -> None:
        """Add one tick, flushing when the buffer fills or the hour turns."""
        if self._closed:
            raise RuntimeError("cannot record after the tick recorder is closed")
        moment = _as_ist(tick.ts)
        hour = f"{moment:%Y-%m-%dT%H}"
        if hour != self._hour and self._buffer:
            # The hour is the partition key, so a buffered row from the previous
            # hour must land before this one starts arriving.
            self.flush()

        row = _tick_row(tick, moment)
        self._append_journal(row)
        self._buffer.append(row)
        self._hour = hour
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> int:
        """Write the buffer out. Returns the rows written."""
        if not self._buffer:
            return 0
        frame = pd.DataFrame(self._buffer).set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index, name="ts")
        written = self.store.write(normalize_frames(frame))
        self.rows_written += written
        self.flushes += 1
        self._buffer.clear()
        self._clear_journal()
        return written

    def _append_journal(self, row: dict) -> None:
        path = tick_journal_path(self.store)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._journal_fd is None:
            self._journal_fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        payload = dict(row)
        payload["ts"] = row["ts"].isoformat()
        encoded = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()
        view = memoryview(encoded)
        while view:
            view = view[os.write(self._journal_fd, view) :]
        self._journal_writes += 1
        if self._journal_writes >= self.journal_sync_every:
            os.fdatasync(self._journal_fd)
            self._journal_writes = 0

    def _clear_journal(self) -> None:
        if self._journal_fd is not None:
            os.fdatasync(self._journal_fd)
            os.close(self._journal_fd)
            self._journal_fd = None
        tick_journal_path(self.store).unlink(missing_ok=True)
        self._journal_writes = 0

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            if self._journal_fd is not None:
                os.fdatasync(self._journal_fd)
                os.close(self._journal_fd)
                self._journal_fd = None
            if self._lock_fd is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = None
            self._closed = True

    def describe(self) -> str:
        coverage = self.store.coverage()
        return (
            f"{self.store.instrument}: {self.rows_written:,} ticks written, "
            f"{self.rows_recovered:,} recovered, {self.flushes} flushes, {self.buffered} buffered"
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


def tick_journal_path(store: PartitionedStore) -> Path:
    """The write-ahead journal holding ticks not checkpointed to parquet yet."""
    return store.base / ".pending.jsonl"


def recover_tick_journal(store: PartitionedStore) -> int:
    """Checkpoint a previous process's complete journal records into parquet.

    A live recorder holds an exclusive lock for its lifetime. If another process
    is actively recording this instrument, its journal is left alone.
    """
    if not tick_journal_path(store).exists():
        return 0
    lock_fd = _lock_journal(store, blocking=False)
    if lock_fd is None:
        return 0
    try:
        return _recover_tick_journal_locked(store)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _recover_tick_journal_locked(store: PartitionedStore) -> int:
    path = tick_journal_path(store)
    if not path.exists() or path.stat().st_size == 0:
        return 0

    rows: list[dict] = []
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
                row["ts"] = _as_ist(row["ts"])
                rows.append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # A process can stop halfway through its final append. Earlier
                # newline-terminated records remain valid and are recovered.
                continue

    if rows:
        frame = pd.DataFrame(rows).set_index("ts")
        store.write(normalize_frames(frame))
    path.unlink(missing_ok=True)

    if rows:
        entry = store.manifest.dataset(store.dataset, store.key)
        recovered = int(entry.get("recovered_rows", 0) or 0) + len(rows)
        store.manifest.update(
            store.dataset,
            store.key,
            recovered_rows=recovered,
            last_recovered_at=datetime.now(IST).isoformat(timespec="seconds"),
            journal_pending=0,
        )
    return len(rows)


def _lock_journal(store: PartitionedStore, blocking: bool) -> int | None:
    path = store.base / ".journal.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, operation)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _tick_row(tick: Tick, moment: pd.Timestamp) -> dict:
    greeks = {str(name): _number(value) for name, value in (tick.greeks or {}).items()}
    row = {
        "ts": moment,
        "instrument_key": tick.instrument_key or "",
        "ltp": _number(tick.ltp),
        "ltq": int(tick.ltq),
        "close_prev": _number(tick.close_prev),
        "bid_p": _number(tick.bid_p),
        "bid_q": int(tick.bid_q),
        "ask_p": _number(tick.ask_p),
        "ask_q": int(tick.ask_q),
        "volume": int(tick.volume_traded),
        "atp": _number(tick.avg_traded_price),
        "oi": _number(tick.open_interest),
        "buy_qty": _number(tick.total_buy_qty),
        "sell_qty": _number(tick.total_sell_qty),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "iv": greeks.get("iv"),
        "greeks_json": json.dumps(greeks, separators=(",", ":"), sort_keys=True),
    }
    identity = dict(row)
    identity["ts"] = moment.isoformat()
    encoded = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    row["event_id"] = hashlib.blake2s(encoded, digest_size=12).hexdigest()
    return row


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_ist(moment) -> pd.Timestamp:
    stamp = pd.Timestamp(moment)
    if stamp.tzinfo is None:
        return stamp.tz_localize(IST)
    return stamp.tz_convert(IST)


__all__ = [
    "CHAIN_SAMPLE_SECONDS",
    "DEFAULT_BUFFER",
    "DEFAULT_JOURNAL_SYNC",
    "CHAIN",
    "ChainRecorder",
    "HOUR",
    "TICKS",
    "TickRecorder",
    "recover_tick_journal",
    "tick_journal_path",
]
