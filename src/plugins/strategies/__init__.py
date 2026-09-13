"""Strategy packs and the catalog that collects them.

A strategy says one thing: given the bars and the feature matrix, how convinced
am I that the next move is up or down, on a scale from -1 to +1. Strategies are
**vectorised** — a pack returns a conviction *series* for every bar at once, not a
decision per bar in a loop. That is what makes a backtest a single pass over a
Series, live inference ``.iloc[-1]``, and the two literally the same code, so a
strategy that looks good in research is the strategy running live rather than a
re-implementation of it.

Packs ship under this package:

* ``trend``      — EMA, supertrend, Donchian, opening range, efficiency
* ``momentum``   — MACD, rate of change, stochastic, activity, order flow
* ``reversion``  — RSI, Bollinger, VWAP, z-score, range fade
* ``volatility`` — squeeze release, breakouts, vol regime, GARCH forecast
* ``ml_forecast`` — the trained model as a strategy, requiring ``forecast``

Each pack advertises one capability per strategy it offers, so the kernel can
answer "is a Donchian breakout available in this build?" without anything
importing the pack.
"""

from .base import (
    CompositeStrategy,
    Strategy,
    StrategyContext,
    gaussian_bell,
    require_columns,
    squash,
)
from .catalog import DEFAULT_WEIGHTS, StrategyCatalog, StrategyPack

__all__ = [
    "DEFAULT_WEIGHTS",
    "CompositeStrategy",
    "Strategy",
    "StrategyCatalog",
    "StrategyContext",
    "StrategyPack",
    "gaussian_bell",
    "require_columns",
    "squash",
]
