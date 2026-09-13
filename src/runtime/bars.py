"""Bars for a session: from the cache, fetched, or generated — decided in one place.

Three packs have an opinion here and none of them should hold the policy:

* ``history``  knows how to cache and how to ask the broker for what is missing
* ``simulated``  knows how to generate a series
* ``upstox``   knows how to talk to the provider

Choosing between them is a runtime decision, and this is where it is made. The
alternative — each pack degrading on its own — is how a platform ends up
generating synthetic bars inside a cache and reporting them as history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from core.settings import Settings
from kernel import Kernel
from plugins.sources import DataUnavailable


@dataclass(slots=True)
class BarLoader:
    """Loads the bar series the platform will run on."""

    kernel: Kernel
    logger: logging.Logger | None = None

    @property
    def settings(self) -> Settings:
        return self.kernel.settings

    def load(
        self,
        days: int = 120,
        refresh: bool = True,
        quiet: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Bars for the session, with the fallback chain made explicit.

        Online: the cache, topped up from the provider.
        Offline, or no credentials: the cache if it holds anything, otherwise a
        generated series which is then cached so the next run is reproducible.
        """
        history = self.kernel.capability("history")

        if not offline:
            try:
                bars = history.load_history(days=days, refresh=refresh, quiet=quiet)
                if not bars.empty:
                    return bars
                self._warn(quiet, "no bars in the cache and nothing new to fetch")
            except DataUnavailable as exc:
                self._warn(quiet, f"{exc} Falling back to generated data.")

        return self.load_offline(days=days, quiet=quiet)

    def load_offline(self, days: int = 120, quiet: bool = False) -> pd.DataFrame:
        """Cached bars if any, otherwise generated and cached."""
        history = self.kernel.capability("history")
        cached = history.load_cached()
        if not cached.empty:
            self._warn(quiet, f"using {len(cached):,} cached bars")
            return cached

        market = self.kernel.build("source:simulated", days=days)
        self._warn(quiet, f"generating {days} sessions of synthetic bars")
        frame = market.candles(days=days)
        return history.seed(frame)

    def _warn(self, quiet: bool, message: str) -> None:
        if not quiet:
            print(message)


__all__ = ["BarLoader"]
