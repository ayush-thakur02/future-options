"""The projection pack: the next few candles, rebuilt continuously.

Provides the ``projection`` capability. Unlike the ML forecaster, which produces a
calibrated probability per horizon once a bar closes, this produces a *path* — the
next three candles as OHLC bars, anchored on the live price and refreshed every
second, so the projected candles lean and breathe with the tape.

It is a separate plugin because it answers a different question with a different
latency budget. A build that wants a slower, purely model-driven view can disable
this pack and keep the ML forecasts; a build that wants the path without any
trained artifacts is served by this pack alone, which is the default offline.
"""

from .candles import (
    DEFAULT_BARS_AHEAD,
    atr_of,
    project_candles,
    projected_frame,
    projection_summary,
    volatility_scale,
)
from .forecaster import ProjectionForecaster
from .momentum import MicroMomentum
from .tracker import HorizonStats, ProjectionScore, ProjectionTracker

__all__ = [
    "DEFAULT_BARS_AHEAD",
    "HorizonStats",
    "MicroMomentum",
    "ProjectionForecaster",
    "ProjectionScore",
    "ProjectionTracker",
    "atr_of",
    "project_candles",
    "projected_frame",
    "projection_summary",
    "volatility_scale",
]
