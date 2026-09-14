"""The Upstox account-backed adapter.

One object holding the credential, the REST client, and the streaming feed, so a
plugin that needs Upstox data depends on a single capability (``broker``) instead
of on three modules. The REST client is built lazily and cached: a process that
only replays cached bars never opens a connection.
"""

from __future__ import annotations

from core.settings import Settings

from .. import DataUnavailable
from .auth import AuthenticationError, TokenStore, clean_token, resolve_token
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
        self._active_token: str | None = None
        self.auth_status = "not checked"

    # ------------------------------------------------------------- credentials

    @property
    def token(self) -> str | None:
        """An explicit env token wins; otherwise the stored one, if unexpired."""
        return self._active_token or resolve_token(self.settings.credentials.access_token, self.settings.token_path)

    def validate_auth(self) -> str:
        """Validate env first; a rejected env value must not shadow a saved login.

        Only 401 permits fallback. Rate limits and network faults are not evidence
        that a credential is bad. No token or signed redirect is logged.
        """
        candidates = [
            ("active", self._active_token),
            (".env/environment", clean_token(self.settings.credentials.access_token)),
            ("saved login", clean_token(TokenStore(self.settings.token_path).load())),
        ]
        seen = set()
        rejected = False
        for source, token in candidates:
            if not token or token in seen:
                continue
            seen.add(token)
            with UpstoxREST(token) as client:
                try:
                    client.ltp(self.settings.instrument_key)
                except AuthenticationError:
                    rejected = True
                    continue
            self._active_token = token
            if self._rest is not None and self._rest.access_token != token:
                self.close()
            self.auth_status = f"authenticated via {source}"
            if rejected:
                self.auth_status += " (rejected token skipped)"
            return token
        raise AuthenticationError(
            "No valid Upstox token. Log in with plugins.sources.upstox.auth.interactive_login; "
            "API key/secret alone cannot stream prices."
        )

    @property
    def is_configured(self) -> bool:
        """Whether live access is possible right now. Never raises."""
        return self.token is not None

    def require_token(self) -> str:
        token = self.token
        if not token:
            raise DataUnavailable(
                "No Upstox access token. Log in with plugins.sources.upstox.auth.interactive_login, "
                "set UPSTOX_ACCESS_TOKEN, or run offline to use simulated data."
            )
        return token

    # ---------------------------------------------------------------- services

    def rest(self) -> UpstoxREST:
        """The REST client, built on first use."""
        if self._rest is None:
            self._rest = UpstoxREST(self._active_token or self.validate_auth())
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
            token_provider=self.validate_auth,
        )

    def close(self) -> None:
        if self._rest is not None:
            self._rest.close()
            self._rest = None

    def __repr__(self) -> str:
        state = "authenticated" if self.is_configured else "no token"
        return f"<UpstoxBroker {self.settings.instrument_key} {state}>"


__all__ = ["UpstoxBroker"]
