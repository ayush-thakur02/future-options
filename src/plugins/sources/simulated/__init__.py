"""Synthetic bars and ticks, for running the whole platform without credentials.

The series is not a random walk with no structure — it has volatility
clustering, intraday volume seasonality, regime shifts, and a weak
mean-reverting component. That is deliberate: a series with no structure at all
would let a broken feature pipeline look identical to a working one, because
"found nothing" and "found nothing because it is broken" would be
indistinguishable.
"""

from .series import BASE_LEVEL, generate_candles
from .ticks import generate_quote, generate_ticks

__all__ = ["BASE_LEVEL", "generate_candles", "generate_quote", "generate_ticks"]
