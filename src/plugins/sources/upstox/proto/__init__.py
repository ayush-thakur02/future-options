"""Generated Upstox feed stubs.

Import the stubs through this package rather than the module inside it::

    from plugins.sources.upstox.proto import MarketDataFeed_pb2 as pb

The generated ``MarketDataFeed_pb2`` declares a dependency on
``google/protobuf/wrappers.proto`` (the ``DoubleValue`` used by ``LTPC.iep``) but
the checked-in stub never imports ``wrappers_pb2``. Protobuf resolves imports
through its descriptor pool at import time, so the module raises::

    Couldn't build proto file into descriptor pool: Depends on file
    'google/protobuf/wrappers.proto', but it has not been loaded

Importing the dependency here fixes it without hand-editing a generated file,
which means the stub can be regenerated at any time. The break was latent until
the plugin loader — which imports every plugin, as it must — imported the feed.
The live Upstox path would have failed on the first connection with the same
error.
"""

from __future__ import annotations

from google.protobuf import wrappers_pb2 as _wrappers_pb2  # noqa: F401  (descriptor dependency)

from . import MarketDataFeed_pb2 as pb

__all__ = ["pb"]
