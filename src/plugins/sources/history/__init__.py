"""The local store: candles, the tape, and the chain, partitioned on disk.

* :mod:`~plugins.sources.history.partitions` — the store itself
* :mod:`~plugins.sources.history.manifest`   — what it holds, without a scan
* :mod:`~plugins.sources.history.recorder`   — writing the live tape as it arrives
* :mod:`~plugins.sources.history.resample`   — session-aware OHLCV aggregation
* :class:`~plugins.sources.history.service.HistorySource` — cache + fetch policy
"""

from .manifest import StoreManifest
from .partitions import (
    CANDLES,
    CHAIN,
    DAY,
    HOUR,
    TICKS,
    PartitionedStore,
    normalize_frames,
    safe_name,
)
from .recorder import ChainRecorder, TickRecorder
from .resample import align_to_session, bar_timestamps, resample_ohlcv
from .service import HistorySource

__all__ = [
    "CANDLES",
    "CHAIN",
    "DAY",
    "HOUR",
    "TICKS",
    "ChainRecorder",
    "HistorySource",
    "PartitionedStore",
    "StoreManifest",
    "TickRecorder",
    "align_to_session",
    "bar_timestamps",
    "normalize_frames",
    "resample_ohlcv",
    "safe_name",
]
