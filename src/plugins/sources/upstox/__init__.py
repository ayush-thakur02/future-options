"""Upstox adapter: credentials, REST, and the live v3 WebSocket feed.

This pack is the only place that knows Upstox exists. That matters for the rest
of the tree: the history store asks the ``broker`` capability for candles rather
than importing a REST client, so pointing the platform at a different provider
is a matter of writing one pack, not editing four.

It offers:

* :mod:`~plugins.sources.upstox.auth` — OAuth 2.0 flow, token lifecycle
* :mod:`~plugins.sources.upstox.rest` — historical, intraday, quote
* :mod:`~plugins.sources.upstox.feed` — streaming ticks over protobuf
* :class:`UpstoxBroker` — the object a plugin is handed, holding all three
"""

from .auth import TokenStore, interactive_login, resolve_token
from .broker import UpstoxBroker
from .feed import UpstoxFeed, authorize_feed_url
from .rest import UpstoxREST

__all__ = [
    "TokenStore",
    "UpstoxBroker",
    "UpstoxFeed",
    "UpstoxREST",
    "authorize_feed_url",
    "interactive_login",
    "resolve_token",
]
