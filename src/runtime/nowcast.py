"""The refresh clock behind the projected candles.

Ticks drive the anchor price and the tick momentum; this drives the *cadence*.
Keeping the two apart is what makes the projection tick every second on a quiet
tape instead of only when a print arrives — and on an index feed, quiet stretches
of several seconds are normal.

The loop is deliberately dumb: sleep, refresh, publish, repeat. The interesting
decisions — how much of the last twenty seconds to believe, how far to damp the
third bar — belong to the projection pack, not to the clock.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .engine import Engine

DEFAULT_INTERVAL = 1.0


@dataclass
class NowcastStats:
    """What the clock actually managed, for the status line and for tests."""

    refreshes: int = 0
    skipped: int = 0
    last_duration_ms: float = 0.0
    worst_duration_ms: float = 0.0

    @property
    def summary(self) -> str:
        if not self.refreshes:
            return "no refreshes yet"
        return (
            f"{self.refreshes} refreshes, last {self.last_duration_ms:.1f}ms, "
            f"worst {self.worst_duration_ms:.1f}ms"
        )


class NowcastLoop:
    """Refreshes projected paths on a fixed interval.

    Takes anything with ``refresh_projection()`` and ``publish()``: a single
    engine, or a board that refreshes every leg on it. The clock has no business
    knowing how many instruments it is driving.
    """

    def __init__(
        self,
        engine: Engine,
        interval: float = DEFAULT_INTERVAL,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine = engine
        self.interval = max(float(interval), 0.01)
        self.stats = NowcastStats()
        self.log = logger or logging.getLogger("niftypulse.nowcast")
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Refresh until stopped. Never raises: a projection is not worth a crash."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                return
            except TimeoutError:
                pass
            await asyncio.to_thread(self.tick_once)

    def tick_once(self) -> bool:
        """One refresh cycle. Returns whether a path was produced."""
        started = time.perf_counter()
        try:
            produced = self.engine.refresh_projection()
        except Exception as exc:  # noqa: BLE001 — the feed must outlive a bad refresh
            self.log.warning("projection refresh failed: %s", exc)
            self.stats.skipped += 1
            return False

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats.refreshes += 1
        self.stats.last_duration_ms = elapsed_ms
        self.stats.worst_duration_ms = max(self.stats.worst_duration_ms, elapsed_ms)
        if not produced:
            self.stats.skipped += 1
        else:
            self.engine.publish()
        return produced

    def stop(self) -> None:
        self._stop.set()

    def __repr__(self) -> str:
        return f"<NowcastLoop every {self.interval:g}s, {self.stats.summary}>"


__all__ = ["DEFAULT_INTERVAL", "NowcastLoop", "NowcastStats"]
