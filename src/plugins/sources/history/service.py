"""The local candle history.

Cached bars are the source of truth for training and backtesting, and only the
missing tail is ever fetched. That keeps repeated runs cheap and stays well
inside the provider's rate limits — which matters because the current session is
routinely re-pulled.

Two things this pack deliberately does *not* do: it does not import a provider's
REST client (that arrives as the ``broker`` capability), and it does not invent
bars when the cache is empty. Deciding to fall back to generated data is a
runtime policy, not a property of a cache — see :mod:`runtime.bars`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.bars import normalize_candles
from core.calendar import IST
from core.settings import Settings

from .. import DataUnavailable
from .partitions import PartitionedStore


class HistorySource:
    """Candles from the local store, topped up from the broker when available."""

    def __init__(
        self,
        settings: Settings,
        broker: Any | None = None,
        offline: bool = False,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.offline = offline
        self.store = PartitionedStore(
            settings.data_dir,
            settings.instrument_key,
            bar_minutes=settings.bar_minutes,
        )
        # Anyone who ran the single-file store has history in it; importing it on
        # first use is the difference between a migration and a disappearance.
        self.store.migrate_legacy()

    # ------------------------------------------------------------- credentials

    @property
    def access_token(self) -> str | None:
        return getattr(self.broker, "token", None) if self.broker is not None else None

    @property
    def can_fetch(self) -> bool:
        return not self.offline and self.access_token is not None

    def rest(self):
        """The broker's REST client, or a clear explanation of what is missing."""
        if self.broker is None:
            raise DataUnavailable(
                "No broker capability is registered, so history cannot be fetched. "
                "Run offline to use cached or generated bars."
            )
        return self.broker.rest()

    def close(self) -> None:
        if self.broker is not None:
            self.broker.close()

    # ------------------------------------------------------------------ loads

    def load_cached(self) -> pd.DataFrame:
        """Everything in the store. Empty if the store does not exist yet."""
        if not self.store.exists():
            return self.store.load()
        return self.store.load()

    def load_history(
        self,
        days: int = 120,
        refresh: bool = True,
        quiet: bool = False,
    ) -> pd.DataFrame:
        """Return 1-minute candles, fetching anything not already cached.

        Returns an empty frame rather than substituting generated data when there
        is neither a cache nor a token: a cache that silently fabricates its own
        contents would make every later offline run unreproducible.
        """
        self.settings.ensure_dirs()

        cached = self.load_cached()
        if self.offline or not self.access_token:
            if not cached.empty and not quiet:
                print(f"Using {len(cached):,} cached bars from {self.store.path}")
            return cached

        if not refresh and not cached.empty:
            return cached

        return self._load_and_refresh(cached, days, quiet)

    def _merge_fetched(self, fetched: pd.DataFrame) -> pd.DataFrame:
        """Store what was fetched and return everything the store now holds."""
        self.store.write(fetched)
        return self.store.load()

    def seed(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace the cache with ``frame`` and return it.

        Used when a caller decides to run on generated data: the decision to
        generate belongs to the runtime, but writing the cache belongs here.
        """
        self.settings.ensure_dirs()
        self.store.replace(frame)
        return frame

    def _load_and_refresh(self, cached: pd.DataFrame, days: int, quiet: bool) -> pd.DataFrame:
        now = pd.Timestamp.now(tz=IST)
        target_start = (now - pd.Timedelta(days=days)).normalize()

        if cached.empty:
            fetch_start = target_start
        else:
            last = cached.index[-1]
            if last >= now - pd.Timedelta(minutes=5):
                if not quiet:
                    print(f"Cache is current ({len(cached):,} bars)")
                return cached
            fetch_start = max((last + pd.Timedelta(minutes=1)).date(), target_start)

        if fetch_start > now.date():
            return cached

        if not quiet:
            print(f"Fetching 1-minute history from {fetch_start} ...")

        client = self.rest()
        fetched = client.fetch_minute_history(
            self.settings.instrument_key,
            days=(now.date() - fetch_start).days + 1,
        )

        # The current session is intraday-only in the v3 API.
        if fetch_start <= now.date():
            try:
                today = client.fetch_intraday(
                    self.settings.instrument_key, "minutes", self.settings.bar_minutes
                )
                if not today.empty:
                    fetched = pd.concat([fetched, today])
                    fetched = fetched[~fetched.index.duplicated(keep="last")]
                    fetched = normalize_candles(fetched)
            except RuntimeError:
                pass

        merged = self._merge_fetched(fetched)
        if not quiet:
            first, last = self.store.coverage()
            print(f"Store now holds {len(merged):,} bars ({first} -> {last})")
        return merged


__all__ = ["HistorySource"]
