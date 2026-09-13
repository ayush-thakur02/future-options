"""The terminal dashboard: a snapshot in, a full screen out.

Layout is composed with Rich ``Layout`` so panels resize with the terminal, and
the whole frame is rebuilt on every refresh. At a few rows of state the cost is
negligible, and it avoids a class of stale-panel bugs that incremental updates
invite.

The dashboard is a *renderer plugin*, so it is handed a snapshot rather than
reaching into the engine. That is what lets the same panels serve a live feed, a
fast replay, and a test that renders a synthetic frame.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.layout import Layout
from rich.live import Live

from core.types import MarketSnapshot
from plugins.forecasts.projection import ProjectionForecaster

from . import panels

# The chart flexes; the forecast and projection panels are sized to their
# content. Letting them flex instead clips the last horizon off the bottom of the
# table, which is the row a reader is most likely to want.
CHART_RATIO = 0.34
MAX_CHART_HEIGHT = 16
MIN_CHART_HEIGHT = 6
MIDDLE_HEIGHT = 9
BOTTOM_HEIGHT = 9


class TerminalRenderer:
    """Renders a :class:`MarketSnapshot` as a full-screen terminal view."""

    def __init__(
        self,
        symbol: str = "NIFTY 50",
        timeframe: str = "1m",
        refresh: float = 1.0,
        forecaster: ProjectionForecaster | None = None,
        console: Console | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.refresh = float(refresh)
        self.forecaster = forecaster
        self.console = console or Console()
        self._live = None

    # ------------------------------------------------------------- composition

    def build(self, snapshot: MarketSnapshot, status: str = "") -> Layout:
        width = max(self.console.width - 2, 60)
        height = max(self.console.height - 4, 24)

        chart_height = int(min(max(height * CHART_RATIO, MIN_CHART_HEIGHT), MAX_CHART_HEIGHT))
        squeeze = height - chart_height - MIDDLE_HEIGHT - BOTTOM_HEIGHT - 4
        if squeeze < 0:
            chart_height = max(chart_height + squeeze, 4)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="chart", size=chart_height + 2),
            Layout(name="middle", size=MIDDLE_HEIGHT),
            Layout(name="bottom", size=BOTTOM_HEIGHT),
            Layout(name="footer", size=1),
        )
        layout["middle"].split_row(
            Layout(name="forecasts", ratio=3),
            Layout(name="projection", ratio=2),
        )
        layout["bottom"].split_row(
            Layout(name="strategies", ratio=2),
            Layout(name="indicators", ratio=3),
        )

        layout["header"].update(panels.header(snapshot, status, self.timeframe))
        layout["chart"].update(panels.chart(snapshot, width, chart_height, self.timeframe))
        layout["forecasts"].update(panels.forecasts(snapshot))
        layout["projection"].update(panels.projection(snapshot, self.forecaster))
        layout["strategies"].update(panels.signals(snapshot))
        layout["indicators"].update(panels.indicators(snapshot))
        layout["footer"].update(panels.footer(snapshot, status))
        return layout

    def show(self, snapshot: MarketSnapshot, status: str = "") -> None:
        """Print a single frame, without taking over the screen."""
        self.console.print(self.build(snapshot, status))

    @contextmanager
    def live(self) -> Iterator[None]:
        """Take over the screen for the duration of the block.

        Split from the loop on purpose: the session owns the cadence and may be
        driving several tasks, so the renderer exposes "start drawing" and
        "draw this frame" rather than a loop of its own.
        """
        with Live(
            self._blank(),
            console=self.console,
            screen=True,
            refresh_per_second=max(int(1.0 / max(self.refresh, 1e-6)), 4),
            transient=False,
        ) as handle:
            self._live = handle
            try:
                yield
            finally:
                self._live = None

    def live_update(self, snapshot: MarketSnapshot, status: str = "") -> None:
        """Push one frame to the live display. A no-op when not in a live block."""
        if self._live is not None:
            self._live.update(self.build(snapshot, status))

    def _blank(self):
        from rich.console import Group

        return Group()

    def run(self, snapshot_source, status_source=None, refresh: float | None = None) -> None:
        """Render ``snapshot_source()`` until interrupted."""
        interval = self.refresh if refresh is None else float(refresh)
        with self.live():
            try:
                while True:
                    self.live_update(
                        snapshot_source(), status_source() if status_source else ""
                    )
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass


def render_once(
    snapshot: MarketSnapshot,
    symbol: str = "NIFTY 50",
    timeframe: str = "1m",
    forecaster: ProjectionForecaster | None = None,
) -> None:
    """Render one frame to stdout, for `niftypulse snapshot` and for tests."""
    renderer = TerminalRenderer(symbol=symbol, timeframe=timeframe, forecaster=forecaster)
    renderer.show(snapshot)


__all__ = ["TerminalRenderer", "render_once"]
