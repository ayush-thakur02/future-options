"""Synthetic bars and ticks, for running the whole platform without credentials.

The series is not a random walk with no structure — it has volatility
clustering, intraday volume seasonality, regime shifts, and a weak
mean-reverting component. That is deliberate: a series with no structure at all
would let a broken feature pipeline look identical to a working one, because
"found nothing" and "found nothing because it is broken" would be
indistinguishable.

:class:`SimulatedFeed` is the clock-paced stream the dashboard runs on offline;
:func:`generate_ticks` is the same idea as a frame, for headless replay.
"""

from .feed import DEFAULT_TICKS_PER_BAR, SimulatedFeed
from .plugin import SimulatedMarket
from .series import BASE_LEVEL, generate_candles
from .ticks import generate_quote, generate_ticks

__all__ = [
    "BASE_LEVEL",
    "DEFAULT_TICKS_PER_BAR",
    "SimulatedFeed",
    "SimulatedMarket",
    "generate_candles",
    "generate_quote",
    "generate_ticks",
]
