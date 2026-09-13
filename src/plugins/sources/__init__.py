"""Where market data comes from.

Three packs ship here:

* ``upstox``    — the account-backed adapter: OAuth, REST, live v3 WebSocket
* ``simulated`` — a generated series, replayable in real time without credentials
* ``history``   — the local parquet cache, and the only pack that reads from disk

A source is a **leaf**: nothing depends on it, so it declares no capability. Which
one is active is a runtime decision, not a wiring one — an Upstox token decides
it in ``niftypulse dashboard``, and ``--offline`` forces the simulated one. That
is why they carry tags rather than competing for a capability that only one of
them could win.

Anything that *consumes* bars depends on the ``bars`` capability instead, which
the aggregator provides. So a source can be swapped without touching a single
consumer, and a new source needs only to satisfy the two protocols below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from core.types import Tick

TickHandler = Callable[[Tick], None]
StatusHandler = Callable[[dict], None]


class DataUnavailable(RuntimeError):
    """Raised when live data is required but the source is not configured."""


@runtime_checkable
class TickFeed(Protocol):
    """What the runtime requires of a streaming source.

    Both the Upstox WebSocket and the synthetic replay satisfy this, which is why
    the engine can be driven unmodified by either.
    """

    status: str

    async def run(self) -> None:
        """Stream until :meth:`stop` is called or the source is exhausted."""

    def stop(self) -> None:
        """Ask the feed to stop. Must be safe to call from another task."""


@runtime_checkable
class BarSource(Protocol):
    """What the runtime requires of anything that hands over historical bars."""

    def load_history(self, days: int = 120, refresh: bool = True, quiet: bool = False): ...


__all__ = ["BarSource", "DataUnavailable", "StatusHandler", "TickFeed", "TickHandler"]
