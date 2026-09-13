"""Tests for the terminal UI.

Two failures motivated these, both of which rendered as "the chart looks broken"
and neither of which any non-visual test would have caught.

1. ``_render_row`` right-stripped each style run, which deleted the leading blank
   run that right-aligns the chart. Every candle jumped to the left edge.
2. The assembled row was three characters wider than the panel interior, so Rich
   wrapped the price axis onto the following line, printing it over the candles.

The second is the important one to guard: an off-by-N in width arithmetic is
invisible in unit tests of the chart alone and only shows up in a full render.
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest
from rich.console import Console
from rich.text import Text

from niftypulse.core.calendar import IST
from niftypulse.core.types import MarketSnapshot
from niftypulse.data.synthetic import generate_candles
from niftypulse.ui.charts import (
    Grid,
    format_price,
    price_axis,
    probability_bar,
    render_candles,
    sparkline,
)
from niftypulse.ui.dashboard import Dashboard


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


def render_lines(bars, width: int, height: int, status: str = "test") -> list[str]:
    import io

    dash = Dashboard(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=width, height=height)

    buffer = io.StringIO()
    console = Console(width=width, height=height, file=buffer, force_terminal=False)
    console.print(dash.build(snapshot(bars), status))
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

    dash = Dashboard(symbol="NIFTY 50", timeframe="1m")
    dash.console = Console(width=110, height=36)
    buffer = io.StringIO()
    Console(width=110, height=36, file=buffer, force_terminal=False).print(
        dash.build(snapshot(bars), "no models")
    )
    assert "no trained models loaded" in buffer.getvalue()


def test_dashboard_renders_with_empty_candles() -> None:
    import io

    dash = Dashboard(symbol="NIFTY 50", timeframe="1m")
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
