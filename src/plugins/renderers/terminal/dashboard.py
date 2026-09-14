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

from core.types import BoardSnapshot, MarketSnapshot
from plugins.forecasts.projection import ProjectionForecaster

from . import board as board_panels
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
        chart_ratio: float = CHART_RATIO,
        max_chart_height: int = MAX_CHART_HEIGHT,
        min_chart_height: int = MIN_CHART_HEIGHT,
        middle_height: int = MIDDLE_HEIGHT,
        bottom_height: int = BOTTOM_HEIGHT,
        show_model_diagnostics: bool = True,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.refresh = float(refresh)
        self.forecaster = forecaster
        self.console = console or Console()
        self.chart_ratio = min(max(float(chart_ratio), 0.15), 0.7)
        self.max_chart_height = max(int(max_chart_height), 4)
        self.min_chart_height = max(min(int(min_chart_height), self.max_chart_height), 4)
        self.middle_height = max(int(middle_height), 6)
        self.bottom_height = max(int(bottom_height), 6)
        self.show_model_diagnostics = bool(show_model_diagnostics)
        self._live = None
        self.view = "research"
        self.strategy_offset = 0

    # ------------------------------------------------------------- composition

    def build(self, snapshot: MarketSnapshot, status: str = "") -> Layout:
        width = max(self.console.width - 2, 60)
        height = max(self.console.height - 4, 24)

        chart_height = int(
            min(
                max(height * self.chart_ratio, self.min_chart_height),
                self.max_chart_height,
            )
        )
        squeeze = height - chart_height - self.middle_height - self.bottom_height - 4
        if squeeze < 0:
            chart_height = max(chart_height + squeeze, 4)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="chart", size=chart_height + 2),
            Layout(name="middle", size=self.middle_height),
            Layout(name="bottom", size=self.bottom_height),
            Layout(name="footer", size=1),
        )
        layout["middle"].split_row(
            Layout(name="forecasts", ratio=3),
            Layout(name="projection", ratio=2),
        )
        if self.show_model_diagnostics:
            layout["bottom"].split_row(
                Layout(name="strategies", ratio=2),
                Layout(name="model_diagnostics", ratio=2),
                Layout(name="indicators", ratio=2),
            )
        else:
            layout["bottom"].split_row(
                Layout(name="strategies", ratio=2),
                Layout(name="indicators", ratio=3),
            )

        layout["header"].update(panels.header(snapshot, status, self.timeframe))
        layout["chart"].update(panels.chart(snapshot, width, chart_height, self.timeframe))
        layout["forecasts"].update(panels.forecasts(snapshot))
        layout["projection"].update(panels.projection(snapshot, self.forecaster))
        layout["strategies"].update(panels.signals(snapshot))
        if self.show_model_diagnostics:
            layout["model_diagnostics"].update(panels.model_diagnostics(snapshot))
        layout["indicators"].update(panels.indicators(snapshot))
        layout["footer"].update(panels.footer(snapshot, status))
        return layout

    def build_board(self, board: BoardSnapshot, status: str = "") -> Layout:
        """The call/index/put layout: three charts, one verdict panel under them.

        The charts take the space; the verdict and the strategies share the row
        beneath. On a narrow terminal the three charts still fit because each is
        only as wide as its own panel — a leg chart of 30 columns is small, but it
        is the same chart, drawn smaller.
        """
        width = self.console.width
        height = self.console.height
        chart_height = min(16, max(4, height - 29))
        bottom = max(height - chart_height - 12, 6)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="charts", size=chart_height + 8),
            Layout(name="bottom", size=bottom),
            Layout(name="footer", size=1),
        )
        ordered = sorted(board.legs, key=lambda leg: {"CALL": 0, "INDEX": 1, "PUT": 2}.get(leg.label, 3))
        layout["charts"].split_row(
            *[Layout(name=leg.label.lower(), ratio=1) for leg in ordered]
        )
        layout["bottom"].split_row(
            Layout(name="strategies", ratio=3),
            Layout(name="indicators", ratio=2),
        )

        layout["header"].update(board_panels.board_header(board, status, self.timeframe))

        # Split the row the way Rich will, rather than approximating it: give the
        # last panel the remainder. A panel told it is wider than it is right-aligns
        # its chart into the gap, and one told it is narrower wraps its price axis.
        count = max(len(board.legs), 1)
        base = max(width // count, 20)
        for index, leg in enumerate(ordered):
            panel_width = base if index < count - 1 else max(width - base * (count - 1), base)
            layout[leg.label.lower()].update(
                board_panels.leg_panel(leg, panel_width, chart_height, self.timeframe)
            )
        spot_leg = board.spot_leg
        if self.view == "prediction":
            layout["strategies"].update(board_panels.prediction_paths(board))
            layout["indicators"].update(board_panels.algorithm_matrix(board))
        else:
            layout["strategies"].update(
                board_panels.scalp_costs(board)
                if self.view == "costs"
                else board_panels.strategy_matrix(
                    board, offset=self.strategy_offset, limit=max(bottom - 5, 1)
                )
            )
            layout["indicators"].update(
                board_panels.ai_scores(board)
                if self.view == "ai"
                else panels.indicators(spot_leg.snapshot)
                if self.view == "indicators" and spot_leg
                else board_panels.research_scores(board)
            )
        layout["footer"].update(board_panels.board_footer(board, status))
        return layout

    def show_board(self, board: BoardSnapshot, status: str = "") -> None:
        self.console.print(self.build_board(board, status))

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
            auto_refresh=False,
        ) as handle:
            self._live = handle
            try:
                yield
            finally:
                self._live = None

    def live_update(self, snapshot: MarketSnapshot | BoardSnapshot, status: str = "") -> None:
        """Push one frame to the live display. A no-op when not in a live block.

        Takes either shape: a board when the option chain is available, a single
        market when it is not, so the session does not have to care which.
        """
        if self._live is None:
            return
        if isinstance(snapshot, BoardSnapshot):
            self._live.update(self.build_board(snapshot, status), refresh=True)
        else:
            self._live.update(self.build(snapshot, status), refresh=True)

    def scroll_strategies(self, delta: int) -> None:
        self.strategy_offset = max(0, self.strategy_offset + int(delta))

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
