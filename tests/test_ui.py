"""Tests for the terminal renderer.

Three failures motivated these, all of which rendered as "the chart looks broken"
and none of which a non-visual test would have caught.

1. ``render_row`` right-stripped each style run, which deleted the leading blank
   run that right-aligns the chart. Every candle jumped to the left edge.
2. The assembled row was three characters wider than the panel interior, so Rich
   wrapped the price axis onto the following line, printing it over the candles.
3. The projected bars have to join the *same* price scale as the printed ones.
   Drawn on their own scale they would look plausible regardless of what they
   said, which is the one thing a projection must never do.

The width guard is the important one: an off-by-N in width arithmetic is
invisible in unit tests of the chart alone and only shows up in a full render.
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest
from rich.console import Console
from rich.text import Text

from core.calendar import IST
from core.types import Direction, ForecastCandle, MarketSnapshot, Prediction
from plugins.forecasts.projection import project_candles
from plugins.renderers.terminal import TerminalRenderer, render_candles
from plugins.renderers.terminal.canvas import Grid
from plugins.renderers.terminal.widgets import (
    format_price,
    price_axis,
    probability_bar,
    sparkline,
)
from plugins.sources.simulated.series import generate_candles


@pytest.fixture(scope="module")
def bars():
    return generate_candles(days=2, seed=17)


# ------------------------------------------------------------------- grid


def test_grid_preserves_leading_blanks() -> None:
    """Leading spaces are positioning, not padding, and must survive.

    This is the exact bug that broke the chart: stripping them moved every
    candle to the left edge of the panel.
    """
    grid = Grid(width=10, height=2)
    grid.put(0, 6, "X", "red")
    rendered = Text.from_markup(grid.render()[0]).plain
    assert rendered == "      X   "
    assert rendered.index("X") == 6


def test_grid_row_width_is_exact() -> None:
    grid = Grid(width=25, height=3)
    grid.put(1, 20, "#", "green")
    for line in grid.render():
        assert Text.from_markup(line).cell_len == 25


def test_grid_groups_identical_styles() -> None:
    """Runs of one style should collapse into a single markup span."""
    grid = Grid(width=6, height=1)
    for column in range(4):
        grid.put(0, column, "█", "green")
    line = grid.render()[0]
    assert line.count("[green]") == 1
    assert Text.from_markup(line).plain == "████  "


# --------------------------------------------------------------- candles


def test_render_candles_dimensions(bars) -> None:
    lines = render_candles(bars.tail(40), width=60, height=9)
    assert len(lines) == 9
    for line in lines:
        assert Text.from_markup(line).cell_len == 60


def test_render_candles_right_aligns_short_series(bars) -> None:
    """A series shorter than the field must sit against the right edge.

    Newest bar on the right is what a chart reader expects, and it is what a
    missing left-pad destroys.
    """
    window = bars.tail(10)
    width = 40
    lines = render_candles(window, width=width, height=8)

    leftmost = min(
        (index for line in lines for index, char in enumerate(Text.from_markup(line).plain)
         if char != " "),
        default=None,
    )
    assert leftmost is not None, "chart rendered nothing"
    assert leftmost >= width - len(window), (
        f"content starts at column {leftmost}, expected >= {width - len(window)}"
    )


def test_render_candles_fills_width_when_series_is_long(bars) -> None:
    """A series at least as long as the field should span the whole field."""
    width = 30
    lines = render_candles(bars.tail(200), width=width, height=8)
    plain = [Text.from_markup(line).plain for line in lines]
    leftmost = min(i for line in plain for i, char in enumerate(line) if char != " ")
    assert leftmost == 0


def test_render_candles_handles_empty() -> None:
    lines = render_candles(generate_candles(days=1, seed=1).iloc[0:0], width=20, height=4)
    assert len(lines) == 4
    assert all(Text.from_markup(line).plain.strip() == "" for line in lines)


def test_render_candles_handles_flat_series(bars) -> None:
    """A zero-range series must not divide by zero."""
    flat = bars.tail(20).copy()
    for column in ("open", "high", "low", "close"):
        flat[column] = 24_000.0
    lines = render_candles(flat, width=30, height=6)
    assert len(lines) == 6


def test_candle_uses_direction_colour(bars) -> None:
    """Up and down bars must be visually distinguishable."""
    lines = render_candles(bars.tail(60), width=50, height=10)
    joined = "".join(lines)
    assert "bright_green" in joined or "green" in joined
    assert "bright_red" in joined or "red" in joined


# ------------------------------------------------------------ price axis


def test_price_axis_uses_the_frame_it_is_given(bars) -> None:
    """The axis must describe the same bars as the chart beside it.

    An earlier version tailed internally, so once the two windows diverged the
    labels described different prices than the candles.
    """
    short = bars.tail(10)
    long = bars.tail(400)
    short_labels = price_axis(short, height=6, width=11)
    long_labels = price_axis(long, height=6, width=11)
    assert short_labels != long_labels


def test_price_axis_top_and_bottom_match_range(bars) -> None:
    window = bars.tail(50)
    labels = price_axis(window, height=8, width=13)
    top = float(labels[0].replace(",", ""))
    bottom = float(labels[-1].replace(",", ""))
    assert top == pytest.approx(float(window["high"].max()), abs=1.0)
    assert bottom == pytest.approx(float(window["low"].min()), abs=1.0)
    assert top > bottom


def test_price_axis_width_is_consistent(bars) -> None:
    for label in price_axis(bars.tail(30), height=7, width=14):
        assert len(label) == 14


# --------------------------------------------------------------- widgets


def test_probability_bar_fills_proportionally() -> None:
    assert probability_bar(0.5, width=10).count("█") == 5
    assert probability_bar(0.9, width=10).count("█") == 9
    assert probability_bar(0.0, width=10).count("█") == 0


def test_probability_bar_clamps_out_of_range() -> None:
    assert probability_bar(1.5, width=10).count("█") == 10
    assert probability_bar(-0.5, width=10).count("█") == 0


def test_sparkline_handles_flat_series() -> None:
    assert sparkline([5.0] * 20, width=10) != ""


def test_sparkline_respects_width() -> None:
    assert len(sparkline(range(500), width=30)) == 30


def test_format_price_thousands_separator() -> None:
    assert format_price(24000.5) == "24,000.50"


# ------------------------------------------------------------- dashboard


def snapshot(bars) -> MarketSnapshot:
    window = bars.tail(120)
    return MarketSnapshot(
        ts=datetime.now(IST),
        symbol="NIFTY 50",
        last_price=float(window["close"].iloc[-1]),
        prev_close=float(window["close"].iloc[0]),
        candles=window,
    )


def render_lines(
    bars,
    width: int,
    height: int,
    status: str = "test",
    projections: list[ForecastCandle] | None = None,
) -> list[str]:
    import io

    frame = snapshot(bars)
    if projections:
        frame.projections = projections

    dash = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=width, height=height)

    buffer = io.StringIO()
    console = Console(width=width, height=height, file=buffer, force_terminal=False)
    console.print(dash.build(frame, status))
    return buffer.getvalue().splitlines()


def chart_panel_lines(lines: list[str]) -> list[str]:
    """The content rows of the candle panel, borders excluded."""
    start = next(
        (i for i, line in enumerate(lines) if "candles" in line and line.startswith("╭")),
        None,
    )
    assert start is not None, "chart panel not found"

    content: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("╰"):
            break
        if line.startswith("│"):
            content.append(line)
    return content


PRICE_LABEL = re.compile(r"\d{2},\d{3}\.\d{2}")


@pytest.mark.parametrize(("width", "height"), [(120, 44), (100, 32), (80, 24), (200, 60)])
def test_chart_axis_is_on_every_candle_row(bars, width: int, height: int) -> None:
    """Every candle row must carry its price label on the same line.

    This is the guard that matters, and a plain width check would miss it: the
    buggy row was 115 characters inside a 118-character console, so nothing
    exceeded the *console*. It exceeded the *panel interior*, so Rich wrapped it
    inside the panel and the axis labels landed on their own lines interleaved
    with the candles — which is what made the chart look scrambled.
    """
    rows = chart_panel_lines(render_lines(bars, width, height))
    assert rows, "chart panel had no content rows"

    for index, row in enumerate(rows):
        assert PRICE_LABEL.search(row), (
            f"chart row {index} has no price label on the same line — "
            f"the row overflowed and wrapped: {row[:90]!r}"
        )


@pytest.mark.parametrize(("width", "height"), [(120, 44), (100, 32), (80, 24), (200, 60)])
def test_dashboard_never_exceeds_console_width(bars, width: int, height: int) -> None:
    """No rendered line may be wider than the console."""
    lines = render_lines(bars, width, height)
    assert lines, "dashboard rendered nothing"
    for index, line in enumerate(lines):
        assert len(line) <= width, (
            f"line {index} is {len(line)} chars, console is {width}: {line[:80]!r}"
        )


def test_dashboard_renders_without_models(bars) -> None:
    """The no-model path must render, not raise."""
    import io

    dash = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=110, height=36)
    buffer = io.StringIO()
    Console(width=110, height=36, file=buffer, force_terminal=False).print(
        dash.build(snapshot(bars), "no models")
    )
    assert "no trained models loaded" in buffer.getvalue()


def test_dashboard_shows_model_agreement_card(bars) -> None:
    frame = snapshot(bars)
    frame.predictions = [
        Prediction(
            ts=frame.ts,
            horizon_min=1,
            p_up=0.68,
            direction=Direction.UP,
            confidence=0.36,
            expected_move_bps=8.0,
            hurdle_bps=3.0,
            contributions={"lightgbm": 0.72, "random_forest": 0.64, "logistic": 0.55},
        )
    ]
    import io

    renderer = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    renderer.console = Console(width=130, height=40)
    buffer = io.StringIO()
    Console(width=130, height=40, file=buffer, force_terminal=False).print(
        renderer.build(frame, "live")
    )
    rendered = buffer.getvalue()
    assert "prediction quality" in rendered
    assert "member range" in rendered
    assert "55%" in rendered or "0.55" in rendered


def test_dashboard_renders_with_empty_candles() -> None:
    import io

    dash = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=110, height=36)
    empty = MarketSnapshot(
        ts=datetime.now(IST),
        symbol="NIFTY 50",
        last_price=0.0,
        prev_close=0.0,
        candles=generate_candles(days=1, seed=1).iloc[0:0],
    )
    buffer = io.StringIO()
    Console(width=110, height=36, file=buffer, force_terminal=False).print(
        dash.build(empty, "starting")
    )
    assert "waiting for bars" in buffer.getvalue()


# -------------------------------------------------------------- projections


def projections_for(bars, conviction: float = 0.7) -> list[ForecastCandle]:
    anchor = float(bars["close"].iloc[-1])
    now = bars.index[-1].to_pydatetime()
    return project_candles(bars, anchor=anchor, conviction=conviction, now=now, bars_ahead=3)


def test_projected_candles_are_drawn_in_blue(bars) -> None:
    """Blue is reserved for projected bars; green and red stay for printed ones."""
    lines = render_candles(bars.tail(40), projections_for(bars), width=60, height=10)
    joined = "".join(lines)
    assert "bright_blue" in joined
    assert "bright_green" in joined or "bright_red" in joined


def styles_between(row: Text, start: int, end: int) -> list[str]:
    """The styles covering a range of *visible* characters.

    Slicing the markup string by index does not work: a row of 60 visible
    characters carries well over a hundred characters of tags, so a slice of the
    markup has nothing to do with the columns on screen. The spans do.
    """
    return [
        str(span.style)
        for span in row.spans
        if span.start < end and span.end > start and span.style is not None
    ]


def test_projected_candles_use_no_direction_colour(bars) -> None:
    """A projected bar must not be able to pass for a printed one.

    Colour is the only thing distinguishing the two at a glance, so a rising
    projection drawn green would be actively misleading.
    """
    projections = projections_for(bars, conviction=0.9)
    width = 60
    lines = render_candles(bars.tail(40), projections, width=width, height=10)

    drawn = "".join(Text.from_markup(line).plain[-3:] for line in lines)
    assert drawn.strip(), "nothing was drawn in the projected columns"

    for line in lines:
        row = Text.from_markup(line)
        for style in styles_between(row, width - 3, width):
            assert "green" not in style, f"a projected candle was drawn {style}"
            assert "red" not in style, f"a projected candle was drawn {style}"


def test_chart_reserves_columns_for_the_projection(bars) -> None:
    """The projected bars sit at the right edge, after a divider."""
    window = bars.tail(40)
    lines = render_candles(window, projections_for(bars), width=60, height=10)
    plain = [Text.from_markup(line).plain for line in lines]

    assert any("┊" in row for row in plain), "no divider between actual and projected"
    assert all(row.rstrip() for row in plain if "┊" in row)


def test_chart_without_projections_is_unchanged(bars) -> None:
    """No projection means no empty columns: the printed bars still fill the width."""
    with_none = render_candles(bars.tail(40), None, width=60, height=10)
    with_empty = render_candles(bars.tail(40), [], width=60, height=10)
    assert with_none == with_empty
    assert not any("┊" in Text.from_markup(line).plain for line in with_none)


def test_projection_above_the_range_expands_the_price_scale(bars) -> None:
    """The axis has to cover the projected bars, not just the printed ones.

    An axis labelled only with the recent range while a blue bar sits above it
    would put price outside the labelled range on screen.
    """
    from plugins.renderers.terminal.candles import render_chart

    flat = bars.tail(40).copy()
    for column in ("open", "high", "low", "close"):
        flat[column] = 24_000.0

    anchor = 24_000.0
    now = flat.index[-1].to_pydatetime()
    far = project_candles(flat, anchor=anchor, conviction=1.0, now=now, bars_ahead=3)

    plain = render_chart(flat, far, width=50, height=10)
    assert plain.top >= max(candle.high for candle in far)


def test_projected_candles_are_inside_the_drawn_rows(bars) -> None:
    """Every projected bar must land somewhere on the canvas."""
    from plugins.renderers.terminal.candles import render_chart

    chart = render_chart(bars.tail(40), projections_for(bars, conviction=1.0), width=50, height=10)
    assert 0 <= chart.last_price_row < len(chart.candle_lines)


def test_dashboard_renders_the_projected_bars(bars) -> None:
    snapshot_with_projection = snapshot(bars)
    snapshot_with_projection.projections = projections_for(bars)
    snapshot_with_projection.conviction = 0.42

    import io

    dash = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=110, height=36)
    buffer = io.StringIO()
    Console(width=110, height=36, file=buffer, force_terminal=False).print(
        dash.build(snapshot_with_projection, "live")
    )
    rendered = buffer.getvalue()
    assert "projected" in rendered
    assert "+1" in rendered and "+3" in rendered
    assert "+0.42 view" in rendered
    assert "█ actual" in rendered and "▓ projected" in rendered


@pytest.mark.parametrize(("width", "height"), [(120, 44), (100, 32), (80, 24), (200, 60)])
def test_chart_axis_stays_on_every_row_with_projections(bars, width: int, height: int) -> None:
    """The width guard must still hold when three projected bars are added.

    Those columns are the extra width that would push the axis onto the next
    line, and the projected bars are exactly where it would go unnoticed.
    """
    lines = render_lines(bars, width, height, projections=projections_for(bars))
    rows = chart_panel_lines(lines)
    assert rows, "chart panel had no content rows"
    for index, row in enumerate(rows):
        assert PRICE_LABEL.search(row), (
            f"chart row {index} has no price label on the same line — "
            f"the row overflowed and wrapped: {row[:90]!r}"
        )
