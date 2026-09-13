"""Colours and glyphs, in one place.

Two visual languages are in play on the chart and they must not be confused:

* **actual** candles — green up, red down, a solid body, the convention every
  chart reader already has
* **projected** candles — blue, never green or red, because a projected bar is a
  different kind of object from a printed one

Brightness carries meaning too: the nearest projected bar is the brightest blue
and the furthest the dimmest, which is the visual form of the horizon decay in
the projection itself. Nothing here invents a colour that is not one of those two
languages, so no bar on the chart is ambiguous about which it is.
"""

from __future__ import annotations

UP_STYLE = "bright_green"
DOWN_STYLE = "bright_red"
FLAT_STYLE = "grey62"

WICK_UP = "│"
WICK_DOWN = "│"
BODY = "█"
HALF_BODY = "▄"

# Projected bars: nearest first. Blue throughout, intensity for horizon.
FORECAST_STYLES = ("bright_blue", "blue", "deep_sky_blue1", "steel_blue1")
FORECAST_WICK = "╎"
# Fill density encodes confidence, so a hesitant projection looks hesitant.
FORECAST_BODY = ("░", "▒", "▓")
CONFIDENCE_STEPS = (0.25, 0.55)

# The boundary between what happened and what is projected.
NOW_DIVIDER = "┊"
NOW_STYLE = "grey42"

# Shade used for the wick of an actual candle. Tinting it keeps direction
# readable in the rows where the body is a single character tall, which on a
# dense chart is most of them.
DIM_STYLES = {"bright_green": "green", "bright_red": "red"}


def dim(style: str) -> str:
    """A dimmer shade of a candle's own colour."""
    return DIM_STYLES.get(style, "grey42")


def forecast_style(horizon: int) -> str:
    """Blue for the projected bars, fading with distance."""
    index = max(int(horizon), 1) - 1
    return FORECAST_STYLES[min(index, len(FORECAST_STYLES) - 1)]


def forecast_body(confidence: float) -> str:
    """Fill density for a projected body, from its confidence."""
    for index, threshold in enumerate(CONFIDENCE_STEPS):
        if confidence < threshold:
            return FORECAST_BODY[index]
    return FORECAST_BODY[-1]


def change_style(value: float) -> str:
    if value > 0:
        return UP_STYLE
    if value < 0:
        return DOWN_STYLE
    return FLAT_STYLE


def arrow(direction: str) -> str:
    return {"UP": "▲", "DOWN": "▼", "FLAT": "▬"}.get(direction, "▬")


__all__ = [
    "BODY",
    "CONFIDENCE_STEPS",
    "DOWN_STYLE",
    "FLAT_STYLE",
    "FORECAST_BODY",
    "FORECAST_STYLES",
    "FORECAST_WICK",
    "HALF_BODY",
    "NOW_DIVIDER",
    "NOW_STYLE",
    "UP_STYLE",
    "WICK_DOWN",
    "WICK_UP",
    "arrow",
    "change_style",
    "dim",
    "forecast_body",
    "forecast_style",
]
