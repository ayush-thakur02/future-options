"""A simulated feed paced by the wall clock.

Offline mode used to replay as fast as the CPU allowed, which is fine for a
backtest and useless for a dashboard: the whole day flashed past in two seconds
and the projected candles had no seconds to move in.

This paces the ticks so that a bar takes as long as it would in the session —
one minute of wall clock for a one-minute bar at ``speed=1`` — and re-bases the
series onto the current time, so the bars it produces are stamped with the clock
the dashboard is showing. That is what makes the live path and the offline path
the same code: both hand the engine a tick whose timestamp is now.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta

import pandas as pd

from core.calendar import IST
from core.types import Tick
from plugins.sources import StatusHandler, TickHandler

DEFAULT_TICKS_PER_BAR = 12


class SimulatedFeed:
    """Streams a generated series in real time, looping until stopped."""

    def __init__(
        self,
        bars: pd.DataFrame,
        on_tick: TickHandler,
        on_status: StatusHandler | None = None,
        speed: float = 1.0,
        ticks_per_bar: int = DEFAULT_TICKS_PER_BAR,
        bar_minutes: int = 1,
        loop: bool = True,
        close_prev: float = 0.0,
    ) -> None:
        if bars is None or bars.empty:
            raise ValueError("SimulatedFeed needs a non-empty bar series to stream")
        self.bars = bars
        self.on_tick = on_tick
        self.on_status = on_status
        self.speed = max(float(speed), 1e-6)
        self.ticks_per_bar = max(int(ticks_per_bar), 1)
        self.bar_minutes = max(int(bar_minutes), 1)
        self.loop = loop
        self.close_prev = float(close_prev)
        self.status = "idle"
        self.ticks_emitted = 0
        self._stop = asyncio.Event()
        self._last_ts: datetime | None = None

    @property
    def tick_interval(self) -> float:
        """Seconds of wall clock between ticks."""
        return self.bar_minutes * 60.0 / self.ticks_per_bar / self.speed

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------------- streaming

    async def run(self) -> None:
        """Emit ticks until stopped, looping the series when it runs out."""
        self.status = "replaying"
        self._emit_status({"type": "connected", "mode": "simulated", "speed": self.speed})
        interval = self.tick_interval

        try:
            while not self._stop.is_set():
                for tick in self._pass_ticks():
                    if self._stop.is_set():
                        break
                    self.on_tick(tick)
                    self._last_ts = tick.ts
                    self.ticks_emitted += 1
                    # Waiting on the stop event rather than sleeping means a long
                    # interval — five seconds between ticks at real-time speed on
                    # a one-minute bar — stays interruptible.
                    if await self._wait(interval):
                        return
                if not self.loop:
                    break
                self._emit_status({"type": "rewound", "ticks": self.ticks_emitted})
        finally:
            self.status = "stopped"
            self._emit_status({"type": "disconnected"})

    async def _wait(self, interval: float) -> bool:
        """Pause for ``interval``. Returns True if the feed was asked to stop."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=interval)
            return True
        except TimeoutError:
            return False

    def _pass_ticks(self) -> Iterator[Tick]:
        """One pass over the series, stamped so the first one starts now.

        The *first* pass is re-based onto the current clock, so the dashboard
        opens on the present rather than on a date in the past. Later passes
        continue from where the last one ended instead of jumping back: a tick
        stream that moves backwards in time is not something the aggregator is
        built to survive, and a simulated market that visibly rewinds would be a
        worse lie than one that simply carries on.

        A generator, not a list: a fifteen-day series is over five thousand bars,
        and materialising every tick of a pass would allocate an object per tick
        for the whole run.
        """
        interval = self.tick_interval
        if self._last_ts is None:
            start = datetime.now(IST).replace(microsecond=0)
        else:
            start = self._last_ts + timedelta(seconds=interval)

        for index, (_, bar) in enumerate(self.bars.iterrows()):
            path = _intra_bar_path(bar, self.ticks_per_bar)
            for step, price in enumerate(path):
                offset = (index * self.ticks_per_bar + step) * interval
                yield Tick(
                    ts=start + timedelta(seconds=offset),
                    ltp=float(price),
                    ltq=1,
                    close_prev=self.close_prev,
                )

    def _emit_status(self, payload: dict) -> None:
        if self.on_status is not None:
            self.on_status(payload)

    def __repr__(self) -> str:
        return f"<SimulatedFeed {len(self.bars)} bars at {self.speed:g}x, interval {self.tick_interval:.2f}s>"


def _intra_bar_path(bar, steps: int) -> list[float]:
    """A plausible price path from open to close, touching the bar's extremes.

    Deterministic, so a replay is reproducible: the same series produces the same
    ticks every time, which is what makes an offline bug findable twice.
    """
    open_price = float(bar["open"])
    close_price = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])
    if steps <= 1:
        return [close_price]

    path: list[float] = []
    for step in range(steps):
        fraction = step / (steps - 1)
        price = open_price + (close_price - open_price) * fraction
        # Bow the path toward whichever extreme the bar reached, so the forming
        # candle shows a high and low before the bar closes.
        swing = (high - max(open_price, close_price)) if close_price >= open_price else (
            min(open_price, close_price) - low
        )
        if swing > 0:
            price += swing * (1.0 - abs(2.0 * fraction - 1.0))
        path.append(price)
    return path


__all__ = ["DEFAULT_TICKS_PER_BAR", "SimulatedFeed"]
