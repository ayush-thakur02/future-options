"""The Upstox account-backed adapter.

One object holding the credential, the REST client, and the streaming feed, so a
plugin that needs Upstox data depends on a single capability (``broker``) instead
of on three modules. The REST client is built lazily and cached: a process that
only replays cached bars never opens a connection.
"""

from __future__ import annotations

from core.settings import Settings

from .. import DataUnavailable
from .auth import resolve_token
from .feed import StatusHandler, TickHandler, UpstoxFeed
from .rest import UpstoxREST

FEED_MODES = frozenset({"ltpc", "full", "full_d30", "option_greeks"})


class UpstoxBroker:
    """Credentials, REST access, and the live feed for one Upstox account."""

    def __init__(self, settings: Settings, feed_mode: str = "full") -> None:
        if feed_mode not in FEED_MODES:
            raise ValueError(
                f"unsupported feed mode {feed_mode!r}; expected one of {sorted(FEED_MODES)}"
            )
        self.settings = settings
        self.feed_mode = feed_mode
        self._rest: UpstoxREST | None = None

    # ------------------------------------------------------------- credentials

    @property
    def token(self) -> str | None:
        """An explicit env token wins; otherwise the stored one, if unexpired."""
        return resolve_token(self.settings.credentials.access_token, self.settings.token_path)

    @property
    def is_configured(self) -> bool:
        """Whether live access is possible right now. Never raises."""
        return self.token is not None

    def require_token(self) -> str:
        token = self.token
        if not token:
            raise DataUnavailable(
                "No Upstox access token. Run `niftypulse login`, set "
                "UPSTOX_ACCESS_TOKEN, or run offline to use simulated data."
            )
        return token

    # ---------------------------------------------------------------- services

    def rest(self) -> UpstoxREST:
        """The REST client, built on first use."""
        if self._rest is None:
            self._rest = UpstoxREST(self.require_token())
        return self._rest

    def feed(
        self,
        on_tick: TickHandler,
        on_status: StatusHandler | None = None,
        instrument_keys: list[str] | None = None,
        mode: str | None = None,
    ) -> UpstoxFeed:
        """A live feed wired to ``on_tick``, ready to be awaited."""
        return UpstoxFeed(
            access_token=self.require_token(),
            instrument_keys=instrument_keys or [self.settings.instrument_key],
            on_tick=on_tick,
            on_status=on_status,
            mode=mode or self.feed_mode,
        )

    def close(self) -> None:
        if self._rest is not None:
            self._rest.close()
            self._rest = None

    def __repr__(self) -> str:
        state = "authenticated" if self.is_configured else "no token"
        return f"<UpstoxBroker {self.settings.instrument_key} {state}>"


__all__ = ["UpstoxBroker"]
