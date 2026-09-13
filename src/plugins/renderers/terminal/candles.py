"""The candlestick chart: printed bars in green and red, projected bars in blue.

The two are drawn on one canvas and one price scale, because they only mean
anything together — a projected bar floating in its own range would look
plausible no matter what it said. Sharing the scale also makes the projection
visibly modest: three blue bars sitting just off the last close is an honest
picture of a one-minute forecast, and it is not a coincidence that it looks
nothing like the move the ensemble would need to clear its cost hurdle.

The projected section is right-aligned after a divider, so the boundary between
what happened and what is expected is a column on the chart rather than a claim
in a caption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.types import ForecastCandle
from plugins.forecasts.projection import projected_frame

from .canvas import Grid
from .theme import (
    BODY,
    DOWN_STYLE,
    FORECAST_WICK,
    HALF_BODY,
    NOW_DIVIDER,
    NOW_STYLE,
    UP_STYLE,
    WICK_DOWN,
    WICK_UP,
    dim,
    forecast_body,
    forecast_style,
)


@dataclass(slots=True)
class Chart:
    """A rendered chart: the candle rows, their axis, and what was drawn."""

    candle_lines: list[str]
    axis_lines: list[str]
    window: pd.DataFrame
    projections: list[ForecastCandle]
    top: float
    bottom: float

    @property
    def last_price_row(self) -> int | None:
        if self.window.empty:
            return None
        close = float(self.window["close"].iloc[-1])
        span = self.top - self.bottom
        if span <= 0:
            return None
        return int(np.clip(round((self.top - close) / span * (len(self.candle_lines) - 1)), 0, len(self.candle_lines) - 1))


def render_chart(
    bars: pd.DataFrame,
    projections: list[ForecastCandle] | None = None,
    width: int = 78,
    height: int = 14,
    axis_width: int = 12,
) -> Chart:
    """Draw printed and projected candles on one scale, plus their price axis."""
    projected = projected_frame(projections) if projections else pd.DataFrame()
    empty = pd.DataFrame(columns=["open", "high", "low", "close"])

    if width <= 2 or height <= 2 or (bars.empty and projected.empty):
        blank = [" " * max(width, 1) for _ in range(max(height, 1))]
        return Chart(
            candle_lines=blank,
            axis_lines=[" " * axis_width for _ in range(max(height, 1))],
            window=empty,
            projections=[],
            top=0.0,
            bottom=0.0,
        )

    window, offset, divider = _layout(bars, projected, width)
    scale = _PriceScale(window, projected, height)
    grid = Grid(width, height)

    for index in range(len(window)):
        open_price = float(window["open"].iloc[index])
        close = float(window["close"].iloc[index])
        rising = close >= open_price
        _draw_candle(
            grid,
            scale,
            column=offset + index,
            high=float(window["high"].iloc[index]),
            low=float(window["low"].iloc[index]),
            open_price=open_price,
            close=close,
            style=UP_STYLE if rising else DOWN_STYLE,
        )

    if not projected.empty:
        if divider is not None:
            grid.fill_column(divider, 0, height - 1, NOW_DIVIDER, NOW_STYLE)
        first = width - len(projected)
        for position in range(len(projected)):
            candle = projections[position] if projections else None
            style = forecast_style(int(candle.horizon) if candle else position + 1)
            open_price = float(projected["open"].iloc[position])
            close = float(projected["close"].iloc[position])
            _draw_candle(
                grid,
                scale,
                column=first + position,
                high=float(projected["high"].iloc[position]),
                low=float(projected["low"].iloc[position]),
                open_price=open_price,
                close=close,
                style=style,
                body=forecast_body(float(candle.confidence) if candle else 0.0),
                wick=FORECAST_WICK,
                wick_style=style,
            )

    return Chart(
        candle_lines=grid.render(),
        axis_lines=scale.axis(height, axis_width),
        window=window,
        projections=list(projections or []),
        top=scale.top,
        bottom=scale.bottom,
    )


def chart_budget(width: int, axis_preferred: int = 10, separator: str = " │ ") -> tuple[int, int]:
    """Split a panel's interior between the candles and the price axis.

    Never returns a total wider than the interior. An axis that does not fit
    wraps and prints over the candles on the following line, which is the failure
    this arithmetic exists to prevent — and three panels across one row is
    exactly where it happens.

    The candle field wins a tie. A nine-column chart with labels beside it is
    harder to read than a nineteen-column chart without them, and the panel title
    carries the last price anyway, so the axis is what gives way.

    Returns ``(chart_width, axis_width)``, where an ``axis_width`` of zero means
    the caller should draw the candles alone.
    """
    interior = max(width - 4, 8)
    if interior - len(separator) - axis_preferred >= 12:
        return interior - len(separator) - axis_preferred, axis_preferred
    return max(interior - len(separator), 6), 0


def render_candles(
    bars: pd.DataFrame,
    projections: list[ForecastCandle] | None = None,
    width: int = 78,
    height: int = 14,
) -> list[str]:
    """Just the candle rows, without the axis."""
    return render_chart(bars, projections, width=width, height=height).candle_lines


def _layout(
    bars: pd.DataFrame, projected: pd.DataFrame, width: int
) -> tuple[pd.DataFrame, int, int | None]:
    """Split the width between printed bars, the divider, and projected ones.

    Returns the bars actually drawn, the column they start at, and the divider
    column. Printed bars are right-aligned against the divider, so the newest one
    sits immediately before it — which is where a reader looks for "now".
    """
    projected_slots = len(projected)
    divider = width - projected_slots - 1 if projected_slots else None
    available = max(width - projected_slots - (1 if projected_slots else 0), 1)
    window = bars.tail(available) if not bars.empty else bars
    offset = max(available - len(window), 0)
    return window, offset, divider


class _PriceScale:
    """Maps a price to a grid row, over the range the whole picture spans."""

    def __init__(self, bars: pd.DataFrame, projected: pd.DataFrame, height: int) -> None:
        self.height = height
        tops: list[float] = []
        bottoms: list[float] = []
        for frame in (bars, projected):
            if frame is not None and not frame.empty:
                tops.append(float(np.nanmax(frame["high"].to_numpy(dtype="float64"))))
                bottoms.append(float(np.nanmin(frame["low"].to_numpy(dtype="float64"))))

        top = max(tops) if tops else 0.0
        bottom = min(bottoms) if bottoms else 0.0
        if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
            centre = top if np.isfinite(top) else 0.0
            top, bottom = centre + 1.0, centre - 1.0

        self.top = top
        self.bottom = bottom
        self.span = top - bottom

    def row(self, price: float) -> int:
        fraction = (self.top - price) / self.span
        return int(np.clip(round(fraction * (self.height - 1)), 0, self.height - 1))

    def axis(self, height: int, width: int) -> list[str]:
        if height <= 1:
            return [" " * width for _ in range(max(height, 1))]
        return [
            f"{self.top - (row / (height - 1)) * self.span:>{width},.2f}"
            for row in range(height)
        ]


def _draw_candle(
    grid: Grid,
    scale: _PriceScale,
    column: int,
    high: float,
    low: float,
    open_price: float,
    close: float,
    style: str,
    body: str = BODY,
    wick: str | None = None,
    wick_style: str | None = None,
) -> None:
    rising = close >= open_price
    high_row, low_row = scale.row(high), scale.row(low)
    open_row, close_row = scale.row(open_price), scale.row(close)
    body_top, body_bottom = min(open_row, close_row), max(open_row, close_row)

    # Wick first, so the body overdraws it where the two overlap.
    grid.fill_column(
        column,
        high_row,
        low_row,
        wick if wick is not None else (WICK_UP if rising else WICK_DOWN),
        wick_style if wick_style is not None else dim(style),
    )

    grid.fill_column(column, body_top, body_bottom, body, style)

    # A doji would otherwise vanish; give it a visible hairline in its own fill,
    # so a flat projected bar still reads as projected rather than printed.
    if body_top == body_bottom:
        grid.put(body_top, column, HALF_BODY if body == BODY else body, style)


__all__ = ["Chart", "chart_budget", "render_candles", "render_chart"]
