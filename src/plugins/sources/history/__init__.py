"""The local candle cache, and the gaps-filling policy on top of it.

Cached bars are the source of truth for training and backtesting. Only the
missing tail is fetched, so repeated runs are cheap and stay inside the
provider's rate limits.
"""

from .resample import align_to_session, bar_timestamps, resample_ohlcv
from .service import HistorySource
from .store import CandleStore

__all__ = [
    "CandleStore",
    "HistorySource",
    "align_to_session",
    "bar_timestamps",
    "resample_ohlcv",
]
