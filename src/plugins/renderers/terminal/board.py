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

from datetime import datetime

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
    if spot_leg:
        text.append(f" · {spot_leg.snapshot.research.get('workers', 1)} CPU workers", style="grey62")
    text.no_wrap = True
    text.overflow = "ellipsis"
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
    snap = leg.snapshot
    age = max((datetime.now(IST) - snap.last_tick_ts).total_seconds(), 0) if snap.last_tick_ts else None
    freshness = f"tick {age:.0f}s ago" if age is not None else "awaiting ticks"
    rows.append(Text(f" {snap.source.upper()} · {freshness}", style="bright_yellow" if age is None or age > 30 else "bright_green"))
    for candle in snap.projections[:3]:
        rows.append(Text(f" +{candle.horizon} {candle.ts:%H:%M}  {candle.close:,.2f}  {candle.change_bps:+.1f}bp", style="bright_blue"))
    for _ in range(max(3 - len(snap.projections), 0)):
        rows.append(Text(" waiting for a current market candle", style="grey50"))
    online = snap.research.get("online", {})
    samples = online.get("samples", {}).get(1, 0)
    rows.append(Text(f" learned {samples:,} · new labels {online.get('live_updates', 0):,} · {snap.research.get('compute_ms', 0):.0f}ms/bar", style="grey62"))
    for row in rows:
        row.no_wrap = True
        row.overflow = "ellipsis"

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
    text.append(" 1 results  2 positions  3 indicators  4 AI  j/k scroll  r refresh  q quit", style="bold white")
    if status:
        text.append("   ")
        text.append(status, style="grey50")
    return text


def research_scores(board: BoardSnapshot) -> Panel:
    table = Table(expand=True, box=None, padding=(0, 1), header_style="grey62")
    for name in ("leg", "+bar", "hit %", "W / L / n", "MAE bp"):
        table.add_column(name, justify="left" if name == "leg" else "right")
    for leg in board.legs:
        stats = leg.snapshot.research.get("horizons", {})
        for horizon in (1, 2, 3):
            stat = stats.get(horizon, {})
            n = stat.get("scored", 0)
            wins = stat.get("hits", 0)
            table.add_row(leg.label if horizon == 1 else "", str(horizon),
                          f"{stat.get('hit_rate', 0):.1%}" if n else "—",
                          f"{wins}/{n-wins}/{n}", f"{stat.get('mean_abs_error_bps', 0):.2f}" if n else "—")
    pending = sum(leg.snapshot.research.get("pending", 0) for leg in board.legs)
    missing = sum(leg.snapshot.research.get("missing", 0) for leg in board.legs)
    failures = [(leg.label, score) for leg in board.legs for score in leg.snapshot.research.get("recent", []) if not score["hit"]]
    footer = Text(f" First issued · {pending} pending · {missing} missing bars\n 0.1bp direction deadband · hit rate is not P&L", style="grey62")
    if failures:
        label, score = max(failures, key=lambda item: item[1]["target"])
        footer.append(f"\n Last miss: {label} +{score['horizon']} {score['target']:%H:%M} {score['projected_close']:,.2f} → {score['actual_close']:,.2f}", style="bright_red")
    return Panel(Group(table, footer), title="[grey62]forward results · frozen forecasts[/]", border_style=PANEL_BORDER)


def strategy_matrix(
    board: BoardSnapshot, offset: int = 0, limit: int = 10
) -> Panel:
    table = Table(expand=True, box=None, padding=(0, 1), header_style="grey62")
    table.add_column("strategy", width=20)
    labels = [label for label in ("INDEX", "CALL", "PUT") if board.leg(label)]
    for label in labels:
        table.add_column(label, width=6)
    table.add_column("index condition / 3-bar hits", ratio=1)
    spot = board.spot_leg
    if spot:
        signals = spot.snapshot.signals
        maximum = max(len(signals) - max(limit, 1), 0)
        offset = min(max(offset, 0), maximum)
        for signal in signals[offset : offset + max(limit, 1)]:
            cells = []
            for label in labels:
                leg = board.leg(label)
                item = next((s for s in leg.snapshot.signals if s.strategy == signal.strategy), None) if leg else None
                state = item.meta.get("state", "WAIT") if item else "N/A"
                text = item.direction.value if state == "ACTIVE" else state
                cells.append(Text(text, style=change_style(item.score) if state == "ACTIVE" else "grey50"))
            n = signal.meta.get("scored", 0)
            measured = (
                f" · {signal.meta.get('hits', 0)}/{n} · "
                f"net {signal.meta.get('net_pnl_bps', 0):+.1f}bp · "
                f"trust {signal.meta.get('trust_score', 0):.0%}"
                if n
                else ""
            )
            table.add_row(signal.strategy, *cells, Text(signal.reason + measured, style="grey62", overflow="ellipsis", no_wrap=True))
        position = f"{offset + 1}-{min(offset + limit, len(signals))}/{len(signals)}"
    else:
        position = "0/0"
    return Panel(Group(table, Text(f" {position} · j/k scroll · WAIT = no setup", style="grey62")),
                 title="[grey62]plugin strategies · independent views[/]", border_style=PANEL_BORDER)


def ai_scores(board: BoardSnapshot) -> Panel:
    table = Table(expand=True, box=None, padding=(0, 1), header_style="grey62")
    for name in ("leg", "algorithm", "n", "acc", "roll", "net bp", "DD", "trust"):
        table.add_column(name, justify="left" if name in {"leg", "algorithm"} else "right")
    signal_lines = []
    for leg in board.legs:
        ai = leg.snapshot.research.get("ai", {})
        for index, card in enumerate(ai.get("scorecards", [])):
            table.add_row(
                leg.label if index == 0 else "",
                card["algorithm"],
                str(card["samples"]),
                f"{card['accuracy']:.1%}" if card["samples"] else "—",
                f"{card['rolling_accuracy']:.1%}" if card["samples"] else "—",
                f"{card['net_pnl_bps']:+.1f}",
                f"{card.get('drawdown_bps', 0):.1f}",
                f"{card['trust_score']:.0%}",
            )
        signals = ai.get("signals", {})
        if signals:
            views = " ".join(
                f"+{h}:{item['action']} {item['p_up']:.2f}/{item['trust_score']:.0%}"
                for h, item in sorted(signals.items())
            )
            signal_lines.append(f"{leg.label} {views}")
    footer = Text("\n".join(signal_lines) if signal_lines else "Waiting for the next completed bar", style="grey62")
    footer.append("\nResearch signals only · probability/trust · no orders are placed", style="bright_yellow")
    return Panel(Group(table, footer), title="[grey62]online AI · prequential scorecards[/]", border_style=PANEL_BORDER)


def scalp_costs(board: BoardSnapshot) -> Panel:
    table = Table(expand=True, box=None, padding=(0, 1), header_style="grey62")
    for label in ("leg / lot", "+bar", "AI", "gross ₹", "cost ₹/unit", "net ₹/lot", "read"):
        table.add_column(label)
    for leg in board.option_legs():
        snap = leg.snapshot
        quantity = int(leg.greeks.get("lot_size", 0))
        if not quantity or not snap.projections:
            table.add_row(leg.label, "—", "—", "—", "waiting for contract / forecast")
            continue
        cost = (snap.research.get("fixed_cost_rupees", 40) / quantity
                + snap.last_price * snap.research.get("variable_cost_bps", 25) / 10_000
                + snap.research.get("spread", 0))
        ai = snap.research.get("ai", {}).get("signals", {})
        for horizon, projected in enumerate(snap.projections, start=1):
            action = ai.get(horizon, {}).get("action", "HOLD")
            position = 1.0 if action == "BUY" else -1.0 if action == "SELL" else 0.0
            gross = position * (projected.close - snap.last_price)
            net = (gross - cost) * quantity if position else 0.0
            table.add_row(
                f"{leg.label} / {quantity}" if horizon == 1 else "",
                str(horizon), action, f"{gross:+.2f}" if position else "—", f"{cost:.2f}",
                Text(f"{net:+.2f}" if position else "—", style=change_style(net)),
                "above cost" if net > 0 else "below cost" if position else "no position",
            )
    note = Text(" BUY/SELL are frozen research classifications for premium direction.\n"
                " Cost = configured fixed fees/lot + variable bps + quoted spread.\n"
                " Charges are assumptions; edit config/default.yaml. INDEX is a reference.\n"
                " Values are hypothetical; no fills or realised P&L are claimed.", style="grey62")
    return Panel(Group(table, note), title="[grey62]position scenarios · next 3 premium bars[/]", border_style=PANEL_BORDER)


__all__ = [
    "ai_scores",
    "board_footer",
    "board_header",
    "leg_panel",
    "research_scores",
    "scalp_costs",
    "strategy_matrix",
    "verdicts",
]
