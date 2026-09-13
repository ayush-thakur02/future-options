"""Bars on disk, partitioned so a write is cheap and a read is narrow.

One file per instrument per trading day:

    data/
      manifest.json
      candles/NSE_INDEX_Nifty_50/2026/09/2026-09-11.parquet
      ticks/NSE_INDEX_Nifty_50/2026-09-11/09.parquet

Three properties, each of which the single-file store it replaces got wrong:

* **A write is proportional to the day, not to the history.** Appending a
  session's bars to one big parquet means rewriting every bar ever stored — to add
  375 rows at a year of 1-minute data is 100k rows read, concatenated and written
  back. Here it rewrites one day.
* **A read is proportional to the range asked for.** Yesterday's close needs
  yesterday's file. Scanning the whole store to find it is work nobody asked for,
  and it is what makes a big store feel slow to open.
* **The partition key is the trading date**, so a session's bars can never be
  split across two files or merged with the next morning's — the mistake naive
  time-based resampling makes, and one this store cannot make because the date is
  the path.

Compression is zstd by default: on sorted numeric columns it is roughly 5-10x
smaller than the equivalent CSV and it decompresses far faster than it compresses,
which is the right trade for data that is written once per day and read constantly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from core.bars import empty_frame, normalize_candles
from core.calendar import IST

from .manifest import StoreManifest

CANDLES = "candles"
TICKS = "ticks"
CHAIN = "chain"
DAY = "day"
HOUR = "hour"
COMPRESSION = "zstd"


def normalize_frames(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce any timestamped frame into the store's index convention.

    Used for datasets that are not candles — the tape and the chain have their own
    columns, and forcing them through the OHLCV schema would quietly discard
    everything that makes them worth storing.

    A frame may be indexed by time alone (the tape) or by ``(ts, strike, kind)``
    (the chain). The second case is not a detail: every strike in a chain snapshot
    shares one timestamp, so an index of time alone would collapse a snapshot of
    eighteen strikes into one row and silently lose the other seventeen.
    """
    if frame is None or frame.empty:
        empty = pd.DataFrame(frame.copy() if frame is not None else {})
        empty.index = pd.DatetimeIndex([], tz=IST, name="ts")
        return empty
    out = frame.copy()
    out.index = _as_ist(out.index)
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def index_stamps(index) -> pd.DatetimeIndex:
    """The timestamp level of an index, whether or not it is a MultiIndex."""
    if isinstance(index, pd.MultiIndex):
        stamps = index.get_level_values(0)
        return stamps if isinstance(stamps, pd.DatetimeIndex) else pd.DatetimeIndex(stamps)
    return pd.DatetimeIndex(index)


def safe_name(instrument: str) -> str:
    """An instrument key as a directory name.

    ``NSE_INDEX|Nifty 50`` becomes ``NSE_INDEX_Nifty_50``: pipes and spaces in a
    path are a portability problem, and the mapping is reversible by eye, which is
    what matters when someone is looking at the folder.
    """
    cleaned = str(instrument).replace("|", "_").replace(" ", "_").replace("/", "-")
    return "".join(character if character.isalnum() or character in "_-." else "_" for character in cleaned)


class PartitionedStore:
    """Candle partitions for one instrument."""

    def __init__(
        self,
        root: Path,
        instrument: str,
        bar_minutes: int = 1,
        dataset: str = CANDLES,
        shard: str = DAY,
        compression: str = COMPRESSION,
        normalizer: Callable[[pd.DataFrame], pd.DataFrame] = normalize_candles,
    ) -> None:
        self.root = Path(root)
        self.instrument = str(instrument)
        self.key = safe_name(self.instrument)
        self.bar_minutes = int(bar_minutes)
        self.dataset = dataset
        self.shard = shard
        self.compression = compression
        self.normalizer = normalizer
        self.base = self.root / dataset / self.key
        self.manifest = StoreManifest(self.root / "manifest.json")

    # ------------------------------------------------------------------ paths

    @property
    def legacy_path(self) -> Path:
        """Where the single-file store used to live, for the one-time migration."""
        return self.root / "candles.parquet"

    def path_for(self, moment) -> Path:
        """The partition file a timestamp belongs to.

        A day shard lands in ``<year>/<month>/<date>.parquet``; an hour shard in
        ``<date>/<HH>.parquet``. Both put the key in the path, which is what makes
        a range read a directory decision instead of a file scan.
        """
        day, hour = self._key_for(moment)
        if self.shard == HOUR:
            return self.base / f"{day.isoformat()}" / f"{hour:02d}.parquet"
        return self.base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.isoformat()}.parquet"

    def _key_for(self, moment) -> tuple[date, int]:
        # Accepts a timestamp, a date, or the first entry of a multi-level index —
        # the caller has a group's first key, not always a bare Timestamp.
        if isinstance(moment, tuple):
            moment = moment[0]
        stamp = pd.Timestamp(moment)
        return stamp.date(), (stamp.hour if self.shard == HOUR else 0)

    def _path_key(self, path: Path) -> tuple[date, int] | None:
        if self.shard == HOUR:
            try:
                return date.fromisoformat(path.parent.name), int(path.stem)
            except ValueError:
                return None
        try:
            return date.fromisoformat(path.stem), 0
        except ValueError:
            return None

    def partitions(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> list[Path]:
        """Every partition file overlapping ``[start, end]``, in date order.

        Filtering by the path rather than by reading the files is the whole point
        of the layout: the date is in the name, so the decision needs no I/O.
        """
        if not self.base.exists():
            return []
        files = sorted(self.base.rglob("*.parquet"))
        if start is None and end is None:
            return files

        lower = self._key_for(start) if start is not None else (date.min, 0)
        upper = self._key_for(end) if end is not None else (date.max, 23)
        kept: list[Path] = []
        for path in files:
            key = self._path_key(path)
            if key is None or lower <= key <= upper:
                kept.append(path)
        return kept

    # ------------------------------------------------------------------- read

    def exists(self) -> bool:
        return self.base.exists() and any(self.base.rglob("*.parquet"))

    def load(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        files = self.partitions(start, end)
        if not files:
            return empty_frame()

        frames = [pd.read_parquet(path) for path in files]
        merged = pd.concat(frames) if len(frames) > 1 else frames[0]
        merged = self.normalizer(merged)

        if start is not None or end is not None:
            stamps = index_stamps(merged.index)
            mask = pd.Series(True, index=merged.index)
            if start is not None:
                mask &= stamps >= _localize(start)
            if end is not None:
                mask &= stamps <= _localize(end)
            merged = merged[mask.to_numpy()]
        return merged

    def coverage(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """First and last timestamp, from the manifest when it has them."""
        if self.shard == HOUR:
            first, last = self.manifest.coverage(self.dataset, self.key)
            if first and last:
                return pd.Timestamp(first), pd.Timestamp(last)
        first, last = self.manifest.coverage(self.dataset, self.key)
        if first and last:
            return pd.Timestamp(first), pd.Timestamp(last)
        frame = self.load()
        if frame.empty:
            return None, None
        return frame.index[0], frame.index[-1]

    def row_count(self) -> int:
        rows = self.manifest.row_count(self.dataset, self.key)
        if rows:
            return rows
        return sum(len(pd.read_parquet(path, columns=["close"])) for path in self.partitions())

    def sessions(self) -> int:
        return len(self.partitions())

    # ------------------------------------------------------------------ write

    def write(self, frame: pd.DataFrame) -> int:
        """Merge ``frame`` into the partitions it belongs to. Returns rows stored.

        Only the partitions the incoming bars actually touch are rewritten, and
        each is deduplicated on the index with the incoming row winning — so
        re-fetching an overlapping window is idempotent, which matters because the
        current session is re-pulled constantly.
        """
        if frame is None or frame.empty:
            return 0

        incoming = self.normalizer(frame)
        written = 0
        for _, shard_frame in incoming.groupby(self._shard_series(incoming)):
            path = self.path_for(shard_frame.index[0])
            merged = _merge_partition(path, shard_frame)
            _write_partition(path, merged, self.compression)
            written += len(shard_frame)

        self._refresh_manifest()
        return written

    def replace(self, frame: pd.DataFrame) -> None:
        """Rebuild the store from ``frame``, removing anything it does not contain."""
        for path in self.partitions():
            path.unlink()
        if frame is not None and not frame.empty:
            self.write(frame)
        else:
            self._refresh_manifest()

    def _shard_series(self, frame: pd.DataFrame) -> pd.Series:
        """The partition each row belongs to, as a groupby key."""
        stamps = index_stamps(frame.index)
        if self.shard == HOUR:
            keys = [f"{day}T{hour:02d}" for day, hour in zip(stamps.date, stamps.hour, strict=True)]
        else:
            keys = [str(day) for day in stamps.date]
        return pd.Series(keys, index=frame.index)

    def _refresh_manifest(self) -> None:
        files = self.partitions()
        if not files:
            self.manifest.forget(self.dataset, self.key)
            return
        first: pd.Timestamp | None = None
        last: pd.Timestamp | None = None
        rows = 0
        for path in files:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            stamps = index_stamps(_as_ist(frame.index))
            rows += len(stamps)
            first = stamps[0] if first is None or stamps[0] < first else first
            last = stamps[-1] if last is None or stamps[-1] > last else last

        self.manifest.update(
            self.dataset,
            self.key,
            rows=rows,
            sessions=len(files),
            first=first.isoformat() if first is not None else None,
            last=last.isoformat() if last is not None else None,
            instrument=self.instrument,
            bar_minutes=self.bar_minutes,
        )

    # -------------------------------------------------------------- migration

    def migrate_legacy(self) -> int:
        """Import the single-file store, once, and move it aside.

        The old layout kept every bar in ``data/candles.parquet``. Anyone who has
        been running the platform has one, and silently starting from an empty
        partitioned store would look exactly like losing their history.
        """
        if not self.legacy_path.exists():
            return 0

        frame = pd.read_parquet(self.legacy_path)
        if frame.empty:
            self.legacy_path.rename(self.legacy_path.with_suffix(".parquet.migrated"))
            return 0

        frame.index = _as_ist(frame.index)
        rows = self.write(self.normalizer(frame))
        self.legacy_path.rename(self.legacy_path.with_suffix(".parquet.migrated"))
        return rows

    # ------------------------------------------------------------ diagnostics

    def describe(self) -> str:
        if not self.exists():
            return f"{self.instrument}: no data"
        first, last = self.coverage()
        if first is None or last is None:
            return f"{self.instrument}: {self.row_count():,} rows"
        return (
            f"{self.instrument}: {self.row_count():,} rows across {self.sessions()} shards "
            f"({first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M})"
        )

    def __repr__(self) -> str:
        return f"<PartitionedStore {self.dataset} {self.key} at {self.base}>"


def _merge_partition(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return incoming
    existing = pd.read_parquet(path)
    existing.index = _as_ist(existing.index)
    combined = pd.concat([existing, incoming])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _write_partition(path: Path, frame: pd.DataFrame, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression=compression)


def _as_ist(index):
    """Timestamp level in IST, preserving any extra index levels."""
    if isinstance(index, pd.MultiIndex):
        levels = list(index.levels)
        stamps = pd.DatetimeIndex(levels[0])
        levels[0] = stamps.tz_localize(IST) if stamps.tz is None else stamps.tz_convert(IST)
        rebuilt = pd.MultiIndex(levels=levels, codes=index.codes, names=index.names)
        return rebuilt
    idx = pd.DatetimeIndex(index, name="ts")
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def _localize(stamp: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(stamp)
    if stamp.tz is None:
        return stamp.tz_localize(IST)
    return stamp.tz_convert(IST)


__all__ = [
    "CANDLES",
    "CHAIN",
    "COMPRESSION",
    "DAY",
    "HOUR",
    "PartitionedStore",
    "TICKS",
    "normalize_frames",
    "safe_name",
]
