"""Live terminal dashboard.

Layout is composed with Rich ``Layout`` so panels resize with the terminal. The
whole frame is rebuilt each refresh: at a few rows of state the cost is
negligible, and it avoids a class of stale-panel bugs that incremental updates
invite.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.calendar import IST
from core.types import MarketSnapshot

from .charts import (
    arrow,
    change_style,
    confidence_meter,
    format_price,
    price_axis,
    probability_bar,
    render_candles,
    sparkline,
)

CONSOLE = Console()


class Dashboard:
    """Renders a :class:`MarketSnapshot` as a full-screen terminal view."""

    def __init__(self, symbol: str = "NIFTY 50", timeframe: str = "1m", refresh: float = 1.0) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.refresh = refresh
        self.console = Console()
        self.frame = 0

    # ------------------------------------------------------------- composition

    def build(self, snapshot: MarketSnapshot, status: str = "") -> Layout:
        width = max(self.console.width - 2, 60)
        height = max(self.console.height - 4, 24)

        # The chart flexes; the forecast and signal panels are sized to fit their
        # content. Letting them flex instead clips the longest horizon off the
        # bottom of the forecasts table, which is exactly the row a user is most
        # likely to want.
        chart_height = max(min(int(height * 0.30), 14), 7)
        middle_height = 9
        available = height - chart_height - middle_height - 3
        if available < 8:
            chart_height = max(chart_height - (8 - available), 5)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="chart", size=chart_height + 2),
            Layout(name="middle", size=middle_height),
            Layout(name="bottom", minimum_size=8),
        )
        layout["middle"].split_row(
            Layout(name="predictions", ratio=3),
            Layout(name="signals", ratio=2),
        )
        layout["bottom"].split_row(
            Layout(name="indicators", ratio=3),
            Layout(name="meta", ratio=2),
        )

        layout["header"].update(self._header(snapshot, status))
        layout["chart"].update(self._chart(snapshot, width, chart_height))
        layout["predictions"].update(self._predictions(snapshot))
        layout["signals"].update(self._signals(snapshot))
        layout["indicators"].update(self._indicators(snapshot))
        layout["meta"].update(self._meta(snapshot))
        return layout

    def _header(self, snapshot: MarketSnapshot, status: str) -> Panel:
        change = snapshot.change
        pct = snapshot.change_pct
        style = change_style(change)

        text = Text()
        text.append(f" {snapshot.symbol} ", style="bold white")
        text.append(f"· {self.timeframe} ", style="grey62")
        text.append("  ")
        text.append(f"{format_price(snapshot.last_price)}", style="bold white")
        text.append("  ")
        text.append(f"{change:+,.2f} ({pct:+.2f}%)", style=f"bold {style}")
        text.append("   ")
        text.append(f"regime: {snapshot.regime}", style="bright_cyan")
        if status:
            text.append("   ")
            text.append(status, style="grey62")
        return Panel(Align.left(text), border_style="grey35", padding=(0, 1))

    def _chart(self, snapshot: MarketSnapshot, width: int, height: int) -> Panel:
        bars = snapshot.candles
        if bars is None or bars.empty:
            return Panel(
                Align.center(Text("waiting for bars...", style="grey50")),
                title="[grey62]price[/]",
                border_style="grey35",
            )

        # Budget the row precisely. ``width`` is the panel's outer width; the
        # borders cost 2 and ``padding=(0, 1)`` costs another 2. Each row is
        # candles + a 3-character separator + the axis, so anything wider than
        # the interior wraps in the panel and the axis ends up printed over the
        # candles on the following line. One column is held back as margin.
        separator = " │ "
        axis_width = 12
        interior = width - 4
        chart_width = max(interior - len(separator) - axis_width - 1, 20)

        # The axis and the candles must describe the same bars, so both receive
        # the identical window rather than each applying its own tail.
        window = bars.tail(chart_width)

        candle_lines = render_candles(window, width=chart_width, height=height)
        axis_lines = price_axis(window, height=height, width=axis_width)

        rows = []
        for i in range(height):
            # The chart strings carry Rich markup, so they must be parsed rather
            # than appended literally — appending raw would print the tags and
            # inflate the line width until the price axis is pushed off-screen.
            row = Text.from_markup(candle_lines[i] if i < len(candle_lines) else "")
            row.append(separator, style="grey35")
            row.append(axis_lines[i] if i < len(axis_lines) else "", style="grey62")
            rows.append(row)

        prices = window["close"].to_numpy()
        footer = Text()
        footer.append(f"  last {len(window)} bars  ", style="grey50")
        if len(prices) > 1:
            footer.append(
                sparkline(prices, width=min(chart_width - 24, 60)),
                style=change_style(float(prices[-1] - prices[0])),
            )
        rows.append(footer)

        return Panel(
            Group(*rows),
            title=f"[grey62]{snapshot.symbol} · {self.timeframe} · candles[/]",
            border_style="grey35",
            padding=(0, 1),
        )

    def _predictions(self, snapshot: MarketSnapshot) -> Panel:
        if not snapshot.predictions:
            body = Align.center(
                Text.from_markup(
                    "no trained models loaded\nrun [bold]niftypulse train[/] to enable forecasts",
                    justify="center",
                ),
                style="grey50",
            )
            return Panel(body, title="[grey62]forecasts[/]", border_style="grey35")

        table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
        table.add_column("h", justify="right", width=4)
        table.add_column("view", width=5)
        table.add_column("P(up)", justify="right", width=6)
        table.add_column("distribution", width=12, no_wrap=True)
        table.add_column("exp move", justify="right", width=9)
        table.add_column("vs cost", justify="right", width=9)

        for prediction in sorted(snapshot.predictions, key=lambda p: p.horizon_min):
            style = change_style(prediction.p_up - 0.5)
            # The tradeable decision, not the directional one: a correct forecast
            # of a sub-cost move is still a losing trade, so the cost column
            # carries the emphasis and the direction is secondary.
            clears = prediction.clears_hurdle
            cost_style = "bold bright_green" if clears else "grey50"
            table.add_row(
                f"{prediction.horizon_min}m",
                Text(f"{arrow(prediction.direction.value)}{prediction.direction.value[:1]}", style=style),
                Text(f"{prediction.p_up:.3f}", style=style),
                Text.from_markup(probability_bar(prediction.p_up, width=12)),
                Text(f"{prediction.expected_move_bps:+.1f}bp", style=style),
                Text(
                    f"{prediction.edge_after_cost_bps:+.1f}bp",
                    style=cost_style,
                ),
            )

        tradeable = [p for p in snapshot.predictions if p.clears_hurdle]
        hurdle = snapshot.predictions[0].hurdle_bps if snapshot.predictions else 0.0

        footer = Text()
        footer.append("round trip  ", style="grey50")
        footer.append(f"{hurdle:.2f}bp  ", style="grey70")
        if tradeable:
            best = max(tradeable, key=lambda p: p.edge_after_cost_bps)
            footer.append("tradeable  ", style="grey50")
            footer.append(
                f"{best.horizon_min}m {best.direction.value} (+{best.edge_after_cost_bps:.1f}bp)",
                style="bold bright_green",
            )
        else:
            footer.append("no horizon clears cost", style="bright_yellow")
        return Panel(
            Group(table, footer),
            title="[grey62]scalp forecasts — does the move pay for the trade?[/]",
            border_style="grey35",
        )

    def _signals(self, snapshot: MarketSnapshot) -> Panel:
        if not snapshot.signals:
            return Panel(
                Align.center(Text("no active signals", style="grey50")),
                title="[grey62]strategies[/]",
                border_style="grey35",
            )

        table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
        table.add_column("strategy", ratio=1)
        table.add_column("view", width=5)
        table.add_column("strength", justify="right", width=9)

        for signal in sorted(snapshot.signals, key=lambda s: -s.strength):
            style = change_style(signal.score)
            table.add_row(
                Text(signal.strategy, style="white"),
                Text(f"{arrow(signal.direction.value)}", style=style),
                Text.from_markup(confidence_meter(signal.strength, width=8)),
            )
        return Panel(table, title="[grey62]active strategies[/]", border_style="grey35")

    def _indicators(self, snapshot: MarketSnapshot) -> Panel:
        indicators = snapshot.indicators or {}
        if not indicators:
            return Panel(Align.center(Text("computing...", style="grey50")), border_style="grey35")

        table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
        table.add_column("indicator", ratio=1)
        table.add_column("value", justify="right", width=9)
        table.add_column("read", ratio=1)

        # Each entry is (label, value formatter, interpretation).
        spec: dict[str, tuple[str, object, object]] = {
            "rsi_14": ("RSI(14)", lambda v: f"{v:.1f}", _read_rsi),
            "adx_14": ("ADX(14)", lambda v: f"{v:.1f}", _read_adx),
            "atr_norm": ("ATR norm", lambda v: f"{v * 10_000:.1f}bp", lambda v: ""),
            "macd_hist": ("MACD hist", lambda v: f"{v * 10_000:+.2f}bp", _read_sign),
            "stoch_k": ("Stoch %K", lambda v: f"{v:.1f}", _read_stoch),
            "bb_pct_b": ("Bollinger %B", lambda v: f"{v:.2f}", _read_band),
            "vwap_dist": ("vs VWAP", lambda v: f"{v * 10_000:+.1f}bp", _read_sign),
            "efficiency_ratio_10": ("Efficiency", lambda v: f"{v:.2f}", _read_efficiency),
        }

        for key, (label, formatter, reader) in spec.items():
            if key not in indicators:
                continue
            value = indicators[key]
            try:
                shown = formatter(value)
                read = reader(value)
            except (TypeError, ValueError):
                shown, read = "—", ""
            table.add_row(label, Text(shown, style="white"), Text(read, style="grey62"))
        return Panel(table, title="[grey62]indicator state[/]", border_style="grey35")

    def _meta(self, snapshot: MarketSnapshot) -> Panel:
        lines = Text()
        lines.append("session\n", style="grey50")
        lines.append(f"  {snapshot.ts.astimezone(IST).strftime('%H:%M:%S IST')}\n", style="white")

        if snapshot.predictions:
            hurdle = snapshot.predictions[0].hurdle_bps
            lines.append("\ncost hurdle\n", style="grey50")
            lines.append(f"  {hurdle:.2f} bp round trip\n", style="bright_yellow")

        lines.append("\nmodel members\n", style="grey50")
        if snapshot.predictions:
            for prediction in sorted(snapshot.predictions, key=lambda p: p.horizon_min):
                lines.append(f"  {prediction.horizon_min}m  {_strongest_member(prediction)}\n", style="grey70")
        else:
            lines.append("  —\n", style="grey50")

        lines.append("\n", style="")
        lines.append("q", style="bold white")
        lines.append(" quit   ", style="grey50")
        lines.append("r", style="bold white")
        lines.append(" refresh", style="grey50")

        return Panel(lines, title="[grey62]session[/]", border_style="grey35")

    # -------------------------------------------------------------- run loop

    def run(self, snapshot_source, status_source=None) -> None:
        """Render ``snapshot_source`` until interrupted."""
        with Live(
            self.build(snapshot_source(), status_source() if status_source else ""),
            console=self.console,
            screen=True,
            refresh_per_second=4,
            transient=False,
        ) as live:
            try:
                while True:
                    snapshot = snapshot_source()
                    status = status_source() if status_source else ""
                    live.update(self.build(snapshot, status))
                    import time

                    time.sleep(self.refresh)
            except KeyboardInterrupt:
                pass


def _blend_view(predictions) -> str:
    if not predictions:
        return "no view"
    average = sum(p.p_up for p in predictions) / len(predictions)
    if average > 0.53:
        return f"UP  ({average:.3f} avg)"
    if average < 0.47:
        return f"DOWN ({average:.3f} avg)"
    return f"neutral ({average:.3f} avg)"


def _strongest_member(prediction) -> str:
    contributions = prediction.contributions or {}
    if not contributions:
        return "—"
    name, value = max(contributions.items(), key=lambda item: abs(item[1] - 0.5))
    direction = "up" if value > 0.5 else "down"
    return f"{name} {direction} {value:.2f}"


def _read_rsi(value: float) -> str:
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "neutral"


def _read_adx(value: float) -> str:
    if value >= 40:
        return "strong trend"
    if value >= 25:
        return "trending"
    if value >= 18:
        return "developing"
    return "range-bound"


def _read_stoch(value: float) -> str:
    if value >= 80:
        return "overbought"
    if value <= 20:
        return "oversold"
    return "neutral"


def _read_sign(value: float) -> str:
    if value > 0:
        return "above"
    if value < 0:
        return "below"
    return "flat"


def _read_band(value: float) -> str:
    if value > 1.0:
        return "above upper band"
    if value < 0.0:
        return "below lower band"
    if value > 0.8:
        return "upper half"
    if value < 0.2:
        return "lower half"
    return "mid band"


def _read_efficiency(value: float) -> str:
    if value > 0.5:
        return "clean directional move"
    if value > 0.25:
        return "moderate"
    return "choppy"


def render_once(snapshot: MarketSnapshot, symbol: str = "NIFTY 50", timeframe: str = "1m") -> None:
    dashboard = Dashboard(symbol=symbol, timeframe=timeframe)
    CONSOLE.print(dashboard.build(snapshot))
