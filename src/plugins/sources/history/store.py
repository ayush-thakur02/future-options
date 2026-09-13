"""Local persistence for candles.

Parquet, deduplicated on the timestamp index, so re-fetching an overlapping
window is idempotent — which matters because the current session is routinely
re-pulled. The OHLCV schema itself lives in :mod:`core.bars`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.bars import as_ist_index, empty_frame
from core.calendar import IST


class CandleStore:
    """Append-only OHLCV store backed by parquet.

    Deduplicates on the timestamp index so re-fetching an overlapping window is
    idempotent — important because we routinely re-pull the current session.
    """

    def __init__(self, path: Path, bar_minutes: int = 1) -> None:
        self.path = path
        self.bar_minutes = bar_minutes

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        if not self.path.exists():
            return empty_frame()
        frame = pd.read_parquet(self.path)
        frame.index = as_ist_index(frame.index)
        frame = frame.sort_index()
        if start is not None:
            frame = frame.loc[frame.index >= _localize(start)]
        if end is not None:
            frame = frame.loc[frame.index <= _localize(end)]
        return frame

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Merge ``frame`` into the store and return the merged result."""
        if frame is None or frame.empty:
            return self.load()

        incoming = frame.copy()
        incoming.index = as_ist_index(incoming.index)
        incoming = incoming[~incoming.index.duplicated(keep="last")]

        existing = self.load()
        if existing.empty:
            merged = incoming
        else:
            merged = pd.concat([existing, incoming])
            merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path)
        return merged

    def replace(self, frame: pd.DataFrame) -> None:
        frame = frame.copy()
        frame.index = as_ist_index(frame.index)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_index().to_parquet(self.path)

    def coverage(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        frame = self.load()
        if frame.empty:
            return None, None
        return frame.index[0], frame.index[-1]

    def row_count(self) -> int:
        if not self.path.exists():
            return 0
        return len(pd.read_parquet(self.path, columns=["close"]))


def _localize(ts: pd.Timestamp) -> pd.Timestamp:
    """A timestamp in IST, so a naive bound compares against the stored index."""
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)
