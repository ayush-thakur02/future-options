"""The canonical OHLCV schema.

Every pack that produces bars — the cache, the provider client, the generator —
and every pack that consumes them agrees on this one definition: a frame indexed
by IST bar-open time with the columns below.

It lives in ``core`` because it is vocabulary, not implementation. The
alternative, which the packs started out with, is that the simulated generator
imports the cache's module to normalise its own output, and a provider client
imports it too — so three packs that have nothing to do with each other end up
coupled through a utility function.
"""

from __future__ import annotations

import pandas as pd

from .calendar import IST

CANDLE_COLUMNS = ["open", "high", "low", "close", "volume", "oi"]


def as_ist_index(index) -> pd.DatetimeIndex:
    """Coerce any index into a datetime index in IST."""
    idx = pd.DatetimeIndex(index, name="ts")
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def empty_frame() -> pd.DataFrame:
    """An empty frame with the canonical columns and index."""
    frame = pd.DataFrame(columns=CANDLE_COLUMNS, dtype="float64")
    frame.index = pd.DatetimeIndex([], tz=IST, name="ts")
    return frame


def normalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw OHLCV frame into the canonical schema."""
    if frame.empty:
        return empty_frame()
    out = frame.copy()
    out.index = as_ist_index(out.index)
    for column in CANDLE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    out = out[CANDLE_COLUMNS]
    return out.sort_index()


__all__ = ["CANDLE_COLUMNS", "as_ist_index", "empty_frame", "normalize_candles"]
