"""The panels of the terminal dashboard.

Each panel answers one question, in the order a scalper asks them: what is price
doing, what is projected next, does the forecast pay for the trade, which
strategies agree, and is anything abnormal in the indicator state.

The chart panel is the only one that draws both printed and projected bars, and
it says so in its own legend. Everything else labels a projected number as
projected, because the single most misleading thing a forecasting UI can do is
put an estimate next to a measurement in the same styling.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.calendar import IST
from core.types import MarketSnapshot
from plugins.forecasts.projection import ProjectionForecaster

from .candles import chart_budget, render_chart
from .theme import arrow, change_style, forecast_style
from .widgets import (
    confidence_meter,
    format_price,
    probability_bar,
    projection_meter,
    sparkline,
)

PANEL_BORDER = "grey35"
TITLE_STYLE = "grey62"


def header(snapshot: MarketSnapshot, status: str, timeframe: str) -> Panel:
    change = snapshot.change
    style = change_style(change)

    text = Text()
    text.append(f" {snapshot.symbol} ", style="bold white")
    text.append(f"· {timeframe} ", style="grey62")
    text.append("  ")
    text.append(f"{format_price(snapshot.last_price)}", style="bold white")
    text.append("  ")
    text.append(f"{change:+,.2f} ({snapshot.change_pct:+.2f}%)", style=f"bold {style}")
    text.append("   ")
    text.append(f"regime {snapshot.regime}", style="bright_cyan")
    text.append("   ")
    text.append(snapshot.ts.astimezone(IST).strftime("%H:%M:%S IST"), style="grey62")

    if snapshot.candles is not None and not snapshot.candles.empty:
        closes = snapshot.candles["close"].tail(60).to_numpy()
        if len(closes) > 1:
            text.append("   ")
            text.append(
                sparkline(closes, width=30),
                style=change_style(float(closes[-1] - closes[0])),
            )
    if status:
        text.append("   ")
        text.append(status, style="grey62")
    return Panel(Align.left(text), border_style=PANEL_BORDER, padding=(0, 1))


def chart(snapshot: MarketSnapshot, width: int, height: int, timeframe: str) -> Panel:
    """Printed candles, then the projected ones in blue, on one price scale."""
    bars = snapshot.candles
    projections = snapshot.projections

    if (bars is None or bars.empty) and not projections:
        return Panel(
            Align.center(Text("waiting for bars...", style="grey50")),
            title=f"[{TITLE_STYLE}]price[/]",
            border_style=PANEL_BORDER,
        )

    # Budget the row precisely. ``width`` is the panel's outer width; the borders
    # cost 2 and ``padding=(0, 1)`` another 2. Each row is candles + a separator +
    # the axis, so anything wider than the interior wraps inside the panel and the
    # axis lands on the following line, printed over the candles. One column is
    # held back as margin.
    separator = " │ "
    chart_width, axis_width = chart_budget(width, axis_preferred=10)

    rendered = render_chart(
        bars if bars is not None else bars.iloc[0:0],
        projections,
        width=chart_width,
        height=height,
        axis_width=axis_width,
    )

    rows = []
    for index in range(height):
        line = rendered.candle_lines[index] if index < len(rendered.candle_lines) else ""
        axis = rendered.axis_lines[index] if index < len(rendered.axis_lines) else ""
        # The chart strings carry Rich markup, so they must be parsed rather than
        # appended literally — appending raw would print the tags and inflate the
        # line width until the price axis is pushed off-screen.
        row = Text.from_markup(line)
        if axis_width:
            row.append(separator, style=PANEL_BORDER)
            row.append(axis, style=TITLE_STYLE)
        rows.append(row)

    # No legend row inside the panel: every row here carries an axis label on the
    # same line, and a legend row would be the one row without one. The key to
    # the colours lives in the footer.
    return Panel(
        Group(*rows),
        title=f"[{TITLE_STYLE}]{snapshot.symbol} · {timeframe} · candles and next {len(projections)} projected[/]",
        border_style=PANEL_BORDER,
        padding=(0, 1),
    )


def forecasts(snapshot: MarketSnapshot) -> Panel:
    """The ML view: does the forecast move pay for the round trip?"""
    if not snapshot.predictions:
        body = Align.center(
            Text.from_markup(
                "no trained models loaded\nrun [bold]niftypulse train[/] to enable forecasts",
                justify="center",
            ),
            style="grey50",
        )
        return Panel(body, title=f"[{TITLE_STYLE}]forecasts[/]", border_style=PANEL_BORDER)

    table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
    table.add_column("h", justify="right", width=4)
    table.add_column("view", width=5)
    table.add_column("P(up)", justify="right", width=6)
    table.add_column("distribution", width=12, no_wrap=True)
    table.add_column("exp move", justify="right", width=9)
    table.add_column("vs cost", justify="right", width=9)

    for prediction in sorted(snapshot.predictions, key=lambda p: p.horizon_min):
        style = change_style(prediction.p_up - 0.5)
        # The tradeable decision, not the directional one: a correct forecast of
        # a sub-cost move is still a losing trade, so the cost column carries the
        # emphasis and the direction is secondary.
        cost_style = "bold bright_green" if prediction.clears_hurdle else "grey50"
        table.add_row(
            f"{prediction.horizon_min}m",
            Text(
                f"{arrow(prediction.direction.value)}{prediction.direction.value[:1]}", style=style
            ),
            Text(f"{prediction.p_up:.3f}", style=style),
            Text.from_markup(probability_bar(prediction.p_up, width=12)),
            Text(f"{prediction.expected_move_bps:+.1f}bp", style=style),
            Text(f"{prediction.edge_after_cost_bps:+.1f}bp", style=cost_style),
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
        title=f"[{TITLE_STYLE}]scalp forecasts — does the move pay for the trade?[/]",
        border_style=PANEL_BORDER,
    )


def projection(snapshot: MarketSnapshot, forecaster: ProjectionForecaster | None = None) -> Panel:
    """The projected path itself, bar by bar, with how it has scored so far."""
    projections = snapshot.projections
    if not projections:
        return Panel(
            Align.center(Text("projecting...", style="grey50")),
            title=f"[{TITLE_STYLE}]next candles[/]",
            border_style=PANEL_BORDER,
        )

    table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
    table.add_column("bar", justify="right", width=4)
    table.add_column("at", justify="right", width=6)
    table.add_column("close", justify="right", width=10)
    table.add_column("move", justify="right", width=8)
    table.add_column("confidence", width=9, no_wrap=True)

    for candle in projections:
        style = forecast_style(candle.horizon)
        stamp = candle.ts.astimezone(IST).strftime("%H:%M") if candle.ts else "—"
        table.add_row(
            Text(f"+{candle.horizon}", style=style),
            Text(stamp, style="grey62"),
            Text(f"{format_price(candle.close)}", style=style),
            Text(f"{candle.change_bps:+.1f}bp", style=style),
            Text.from_markup(projection_meter(candle.confidence, width=8)),
        )

    footer = Text()
    footer.append("view  ", style="grey50")
    direction = "up" if snapshot.conviction >= 0 else "down"
    footer.append(f"{direction} {snapshot.conviction:+.2f}  ", style="bright_blue")
    if forecaster is not None:
        stats = forecaster.tracker.stat(1)
        if stats is not None and stats.scored:
            footer.append("scored  ", style="grey50")
            footer.append(
                f"{stats.hit_rate:.0%} hit, {stats.mean_abs_error_bps:.1f}bp error "
                f"over {stats.scored}",
                style="grey70",
            )
        else:
            footer.append("no projection scored yet", style="grey50")
    return Panel(
        Group(table, footer),
        title=f"[{TITLE_STYLE}]projected candles — rebuilt every second[/]",
        border_style=PANEL_BORDER,
    )


def signals(snapshot: MarketSnapshot) -> Panel:
    if not snapshot.signals:
        return Panel(
            Align.center(Text("no active signals", style="grey50")),
            title=f"[{TITLE_STYLE}]strategies[/]",
            border_style=PANEL_BORDER,
        )

    table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
    table.add_column("strategy", ratio=1)
    table.add_column("view", width=5)
    table.add_column("strength", justify="right", width=9)

    for signal in sorted(snapshot.signals, key=lambda item: -item.strength):
        style = change_style(signal.score)
        table.add_row(
            Text(signal.strategy, style="white"),
            Text(f"{arrow(signal.direction.value)}", style=style),
            Text.from_markup(confidence_meter(signal.strength, width=8)),
        )
    return Panel(table, title=f"[{TITLE_STYLE}]active strategies[/]", border_style=PANEL_BORDER)


def indicators(snapshot: MarketSnapshot) -> Panel:
    values = snapshot.indicators or {}
    if not values:
        return Panel(Align.center(Text("computing...", style="grey50")), border_style=PANEL_BORDER)

    table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
    table.add_column("indicator", ratio=1)
    table.add_column("value", justify="right", width=9)
    table.add_column("read", ratio=1)

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
        if key not in values:
            continue
        value = values[key]
        try:
            shown = formatter(value)
            read = reader(value)
        except (TypeError, ValueError):
            shown, read = "—", ""
        table.add_row(label, Text(shown, style="white"), Text(read, style="grey62"))
    return Panel(table, title=f"[{TITLE_STYLE}]indicator state[/]", border_style=PANEL_BORDER)


def footer(snapshot: MarketSnapshot, status: str) -> Text:
    """One line: the key to the chart's colours, what drives the projection, and
    how to leave.
    """
    text = Text()
    text.append(" █ ", style="bright_green")
    text.append("actual  ", style="grey50")
    text.append("▓ ", style="bright_blue")
    text.append("projected  ", style="grey50")
    text.append("│", style="grey35")
    text.append("  next ", style="grey50")
    text.append(f"{len(snapshot.projections)} bars  ", style="bright_blue")
    text.append(f"{snapshot.conviction:+.2f} view  ", style="grey70")
    text.append("│", style="grey35")
    text.append(" q", style="bold white")
    text.append(" quit  ", style="grey50")
    text.append("r", style="bold white")
    text.append(" refresh", style="grey50")
    if status:
        text.append("   ")
        text.append(status, style="grey50")
    return text


def _strongest_member(prediction) -> str:
    contributions = prediction.contributions or {}
    if not contributions:
        return "—"
    name, value = max(contributions.items(), key=lambda item: abs(item[1] - 0.5))
    return f"{name} {'up' if value > 0.5 else 'down'} {value:.2f}"


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


__all__ = [
    "chart",
    "footer",
    "forecasts",
    "header",
    "indicators",
    "projection",
    "signals",
]
