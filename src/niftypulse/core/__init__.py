"""Core vocabulary shared by the kernel and every plugin.

Three modules, no dependencies on anything above them:

* :mod:`~niftypulse.core.types` — the domain types that cross plugin
  boundaries: ``Tick``, ``Signal``, ``Prediction``, ``ForecastCandle``,
  ``MarketSnapshot``, ``Trade``.
* :mod:`~niftypulse.core.settings` — resolved configuration and the cost hurdle.
* :mod:`~niftypulse.core.calendar` — the NSE session clock.

Plugins depend on ``core``; ``core`` depends on nothing but the standard library
and pandas. That single rule is what keeps a plugin graph acyclic no matter how
many plugins are loaded.
"""

from .calendar import IST, SESSION_CLOSE, SESSION_MINUTES, SESSION_OPEN, TradingCalendar, to_ist
from .settings import (
    INDIA_VIX_KEY,
    NIFTY50_KEY,
    NIFTYBANK_KEY,
    Settings,
    UpstoxCredentials,
    load_settings,
    project_root,
)
from .types import (
    Direction,
    ForecastCandle,
    MarketSnapshot,
    Prediction,
    Signal,
    Tick,
    Trade,
)

__all__ = [
    "INDIA_VIX_KEY",
    "IST",
    "NIFTY50_KEY",
    "NIFTYBANK_KEY",
    "SESSION_CLOSE",
    "SESSION_MINUTES",
    "SESSION_OPEN",
    "Direction",
    "ForecastCandle",
    "MarketSnapshot",
    "Prediction",
    "Settings",
    "Signal",
    "Tick",
    "Trade",
    "TradingCalendar",
    "UpstoxCredentials",
    "load_settings",
    "project_root",
    "to_ist",
]
