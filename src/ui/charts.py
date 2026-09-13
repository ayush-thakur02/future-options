"""Text-mode chart rendering.

Everything is drawn onto a character grid and then colourised by grouping runs of
identically styled characters. Emitting Rich markup per character instead would
produce a string many times the size of the visible output and make the dashboard
visibly lag on a busy feed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UP_STYLE = "bright_green"
DOWN_STYLE = "bright_red"
FLAT_STYLE = "grey62"
WICK_UP = "│"
WICK_DOWN = "│"
BODY = "█"
HALF_BODY = "▄"


def format_price(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def format_signed(value: float, decimals: int = 2, suffix: str = "") -> str:
    return f"{value:+,.{decimals}f}{suffix}"


def format_bps(value: float) -> str:
    return f"{value:+.2f}bp"


@dataclass
class Cell:
    char: str
    style: str


class Grid:
    """A character canvas that supports run-length style grouping."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells: list[list[Cell | None]] = [[None] * width for _ in range(height)]

    def put(self, row: int, column: int, char: str, style: str = "") -> None:
        if 0 <= row < self.height and 0 <= column < self.width and char:
            self.cells[row][column] = Cell(char, style)

    def render(self) -> list[str]:
        """Render each row, exactly ``width`` visible characters wide.

        Every cell produces exactly one character, so a row's visible length is
        already correct and no padding is needed. Padding was previously applied
        here to compensate for stripping done in ``_render_row``; now that the
        stripping is gone, the two would fight and push the price axis off the
        panel.
        """
        return [_render_row(row) for row in self.cells]


def _render_row(row: list[Cell | None]) -> str:
    """Render one grid row to markup, grouping runs of identical style.

    Leading blanks are **preserved**. They are what positions the chart inside
    the panel: a shorter series is right-aligned so the newest bar sits at the
    right edge, and stripping the leading run of spaces shifts every candle to
    the far left instead. An earlier version called ``rstrip`` on each run, which
    did exactly that and made the chart look broken.
    """
    parts: list[str] = []
    run_style: str | None = None
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        text = "".join(run)
        parts.append(f"[{run_style}]{text}[/]" if run_style else text)
        run.clear()

    for cell in row:
        if cell is None:
            char, style = " ", None
        else:
            char, style = cell.char, (cell.style or None)
        if style != run_style:
            flush()
            run_style = style
        run.append(char)
    flush()
    return "".join(parts)


def render_candles(
    bars: pd.DataFrame,
    width: int = 78,
    height: int = 14,
    decimals: int = 2,
) -> list[str]:
    """Render OHLC bars as a candlestick chart.

    Each bar occupies one character column. The wick spans the high-low range and
    the body spans open-close, so the shape reads the same way as a graphical
    chart at a fraction of the width.
    """
    if bars.empty or width <= 2 or height <= 2:
        return [" " * max(width, 1) for _ in range(max(height, 1))]

    frame = bars.tail(width).copy()
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    opens = frame["open"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")

    top = np.nanmax(highs)
    bottom = np.nanmin(lows)
    if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
        center = top if np.isfinite(top) else 0.0
        top, bottom = center + 1.0, center - 1.0

    span = top - bottom
    grid = Grid(width, height)

    def to_row(price: float) -> int:
        fraction = (top - price) / span
        return int(np.clip(round(fraction * (height - 1)), 0, height - 1))

    offset = width - len(frame)
    for i in range(len(frame)):
        column = offset + i
        high_row = to_row(highs[i])
        low_row = to_row(lows[i])
        open_row = to_row(opens[i])
        close_row = to_row(closes[i])

        rising = closes[i] >= opens[i]
        style = UP_STYLE if rising else DOWN_STYLE

        body_top, body_bottom = min(open_row, close_row), max(open_row, close_row)

        # Wick first so the body overdraws it where they overlap.
        for row in range(high_row, low_row + 1):
            grid.put(row, column, WICK_UP if rising else WICK_DOWN, _dim(style))

        for row in range(body_top, body_bottom + 1):
            grid.put(row, column, BODY, style)

        # A doji would otherwise vanish; give it a visible hairline.
        if body_top == body_bottom:
            grid.put(body_top, column, HALF_BODY, style)

    return grid.render()


def _dim(style: str) -> str:
    """A dimmer shade of the candle's own colour.

    Rendering every wick in the same neutral grey loses direction information in
    exactly the rows where the body is only one character tall, which is most of
    them on a dense chart. Tinting the wick keeps up and down distinguishable.
    """
    return {"bright_green": "green", "bright_red": "red"}.get(style, "grey42")


def probability_bar(probability: float, width: int = 12) -> str:
    """A filled bar showing a probability, coloured by direction."""
    probability = float(np.clip(probability, 0.0, 1.0))
    filled = int(round(probability * width))
    style = UP_STYLE if probability >= 0.5 else DOWN_STYLE
    return f"[{style}]{'█' * filled}[/][grey30]{'░' * (width - filled)}[/]"


def confidence_meter(confidence: float, width: int = 10) -> str:
    confidence = float(np.clip(confidence, 0.0, 1.0))
    filled = int(round(confidence * width))
    style = "bright_cyan" if confidence > 0.4 else "grey62"
    return f"[{style}]{'▮' * filled}[/][grey30]{'▯' * (width - filled)}[/]"


def sparkline(values, width: int = 40) -> str:
    """Compact trend line using block characters."""
    series = pd.Series(values).dropna()
    if series.empty:
        return ""
    if len(series) > width:
        step = len(series) / width
        series = series.iloc[[int(i * step) for i in range(width)]]

    blocks = "▁▂▃▄▅▆▇█"
    low, high = float(series.min()), float(series.max())
    if high <= low:
        return blocks[0] * len(series)

    scaled = ((series - low) / (high - low) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[index] for index in scaled)


def price_axis(
    bars: pd.DataFrame,
    height: int = 14,
    decimals: int = 2,
    width: int = 11,
) -> list[str]:
    """Price labels aligned to the candle chart rows.

    Uses the full frame it is given, so the caller controls the window. Tailing
    internally would let the axis describe a different set of bars than the chart
    beside it whenever the two lengths diverged.
    """
    if bars.empty or height <= 1:
        return [" " * width for _ in range(max(height, 1))]

    high = float(bars["high"].max())
    low = float(bars["low"].min())
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return [" " * width for _ in range(height)]

    lines = []
    for row in range(height):
        fraction = row / (height - 1)
        price = high - fraction * (high - low)
        lines.append(f"{price:>{width},.{decimals}f}")
    return lines


def change_style(value: float) -> str:
    if value > 0:
        return UP_STYLE
    if value < 0:
        return DOWN_STYLE
    return FLAT_STYLE


def arrow(direction: str) -> str:
    return {"UP": "▲", "DOWN": "▼", "FLAT": "▬"}.get(direction, "▬")
