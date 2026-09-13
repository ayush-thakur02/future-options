"""The board: call, index and put side by side, with the verdict underneath.

Three charts of the same market, one clock, one second apart. They share a
layout but not a price scale — a premium and an index level are different
quantities, and forcing them onto one axis would be the most misleading thing
this screen could do.

Colour stays consistent with the single-instrument view: printed bars in green
and red, projected bars in blue at every leg. The verdict panel is where the
arithmetic lives, and it is deliberately placed under the charts rather than
above them: the requirement is the number a reader should see before the
conclusion, not after.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.calendar import IST
from core.types import BoardSnapshot, LegSnapshot

from .candles import chart_budget, render_chart
from .theme import arrow, change_style
from .widgets import format_price, sparkline

PANEL_BORDER = "grey35"
TITLE_STYLE = "grey62"
ACTION_STYLES = {"LONG": "bold bright_green", "SHORT": "bold bright_red", "FLAT": "grey50"}


def board_header(board: BoardSnapshot, status: str, timeframe: str) -> Panel:
    spot_leg = board.spot_leg
    change = spot_leg.snapshot.change if spot_leg else 0.0
    pct = spot_leg.snapshot.change_pct if spot_leg else 0.0
    style = change_style(change)

    text = Text()
    text.append(f" {board.symbol} ", style="bold white")
    text.append(f"· {timeframe} ", style="grey62")
    text.append("  ")
    text.append(format_price(board.spot), style="bold white")
    text.append("  ")
    text.append(f"{change:+,.2f} ({pct:+.2f}%)", style=f"bold {style}")
    if spot_leg is not None:
        text.append("   ")
        text.append(f"regime {spot_leg.snapshot.regime}", style="bright_cyan")
        text.append("   ")
        text.append(f"view {spot_leg.snapshot.conviction:+.2f}", style="bright_blue")

    expiry = board.chain.get("expiry") if board.chain else None
    if expiry:
        text.append("   ")
        text.append(f"expiry {expiry}", style="grey62")
        text.append("  ")
        text.append(f"PCR {board.chain.get('pcr', 0.0):.2f}", style="grey70")
    text.append("   ")
    text.append(board.ts.astimezone(IST).strftime("%H:%M:%S IST"), style="grey62")
    if status:
        text.append("  ")
        text.append(status, style="grey62")
    return Panel(Align.left(text), border_style=PANEL_BORDER, padding=(0, 1))


def leg_panel(leg: LegSnapshot, width: int, height: int, timeframe: str) -> Panel:
    """One instrument's chart, with its own numbers underneath."""
    bars = leg.snapshot.candles
    separator = " │ "
    chart_width, axis_width = chart_budget(width, axis_preferred=10)

    rendered = render_chart(
        bars if bars is not None else bars.iloc[0:0],
        leg.snapshot.projections,
        width=chart_width,
        height=height,
        axis_width=axis_width,
    )

    rows = []
    for index in range(height):
        line = rendered.candle_lines[index] if index < len(rendered.candle_lines) else ""
        axis = rendered.axis_lines[index] if index < len(rendered.axis_lines) else ""
        row = Text.from_markup(line)
        if axis_width:
            row.append(separator, style=PANEL_BORDER)
            row.append(axis, style=TITLE_STYLE)
        rows.append(row)
    rows.append(_strip(leg))

    return Panel(
        Group(*rows),
        title=f"[{TITLE_STYLE}]{_title(leg, timeframe)}[/]",
        border_style="bright_blue" if leg.snapshot.projections else PANEL_BORDER,
        padding=(0, 1),
    )


def _title(leg: LegSnapshot, timeframe: str) -> str:
    if not leg.is_option:
        return f"{leg.label} · {format_price(leg.snapshot.last_price)} · {timeframe}"
    return f"{leg.label} {leg.strike:,.0f} · {leg.snapshot.last_price:,.2f} · {timeframe}"


def _strip(leg: LegSnapshot) -> Text:
    """The one line of numbers that decides whether the chart matters."""
    text = Text()
    if leg.is_option:
        greeks = leg.greeks
        text.append("  Δ ", style="grey50")
        text.append(f"{greeks.get('delta', 0.0):+.2f}", style="white")
        text.append("  θ/min ", style="grey50")
        text.append(f"{greeks.get('theta_per_minute', 0.0):+.3f}", style="bright_red")
        text.append("  IV ", style="grey50")
        text.append(f"{greeks.get('iv', 0.0):.1%}", style="white")
        text.append("  needs ", style="grey50")
        text.append(f"{greeks.get('breakeven_bps', 0.0):.0f}bp", style="bright_yellow")
    else:
        text.append("  conv ", style="grey50")
        text.append(f"{leg.snapshot.conviction:+.2f}", style="bright_blue")
        closes = leg.snapshot.candles["close"].tail(40).to_numpy() if not leg.snapshot.candles.empty else []
        if len(closes) > 1:
            text.append("  ")
            text.append(sparkline(closes, width=14), style=change_style(float(closes[-1] - closes[0])))
    return text


def verdicts(board: BoardSnapshot) -> Panel:
    """Every leg's requirement and its verdict, arithmetic first."""
    table = Table(expand=True, box=None, header_style="grey50", padding=(0, 1))
    table.add_column("leg", width=13)
    table.add_column("do", width=4)
    table.add_column("needs", justify="right", width=8)
    table.add_column("projected", justify="right", width=10)
    table.add_column("edge", justify="right", width=8)
    table.add_column("why", ratio=1)

    for leg in board.legs:
        verdict = leg.verdict
        if verdict is None:
            continue
        style = ACTION_STYLES.get(verdict.action, "grey50")
        table.add_row(
            Text(f"{verdict.label}{'' if not leg.is_option else f' {leg.strike:,.0f}'}", style="white"),
            Text(f"{arrow('UP' if verdict.projected_move_bps >= 0 else 'DOWN')}{verdict.action[:1]}", style=style),
            Text(f"{verdict.required_move_bps:,.0f}bp", style="bright_yellow"),
            Text(f"{verdict.projected_move_bps:+,.1f}bp", style=change_style(verdict.projected_move_bps)),
            Text(f"{verdict.edge_bps:+,.0f}bp", style=style),
            Text(verdict.reason, style="grey62"),
        )

    footer = Text()
    footer.append("  ", style="")
    footer.append(board.headline or "no verdict", style="bold white" if board.tradeable() else "grey62")
    if board.chain:
        footer.append("   ", style="")
        footer.append(f"CE OI {board.chain.get('call_oi', 0):,.0f}", style="grey50")
        footer.append("  ", style="")
        footer.append(f"PE OI {board.chain.get('put_oi', 0):,.0f}", style="grey50")
    return Panel(
        Group(table, footer),
        title=f"[{TITLE_STYLE}]does the move pay for the position?[/]",
        border_style="bright_yellow" if board.tradeable() else PANEL_BORDER,
    )


def board_footer(board: BoardSnapshot, status: str) -> Text:
    text = Text()
    text.append(" █ ", style="bright_green")
    text.append("actual  ", style="grey50")
    text.append("▓ ", style="bright_blue")
    text.append("projected  ", style="grey50")
    text.append("│", style="grey35")
    legs = " ".join(f"{leg.label} {len(leg.snapshot.projections)}" for leg in board.legs)
    text.append("  next ", style="grey50")
    text.append(f"{legs} bars", style="bright_blue")
    text.append("  │", style="grey35")
    text.append(" q", style="bold white")
    text.append(" quit  ", style="grey50")
    text.append("r", style="bold white")
    text.append(" refresh", style="grey50")
    if status:
        text.append("   ")
        text.append(status, style="grey50")
    return text


__all__ = ["board_footer", "board_header", "leg_panel", "verdicts"]
