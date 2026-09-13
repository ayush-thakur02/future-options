"""Small readouts: formatting, bars, meters, sparklines and the price axis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .theme import DOWN_STYLE, UP_STYLE


def format_price(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def format_signed(value: float, decimals: int = 2, suffix: str = "") -> str:
    return f"{value:+,.{decimals}f}{suffix}"


def format_bps(value: float) -> str:
    return f"{value:+.2f}bp"


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


def projection_meter(confidence: float, width: int = 8) -> str:
    """The same idea in blue, so a projected bar's confidence reads as projected."""
    confidence = float(np.clip(confidence, 0.0, 1.0))
    filled = int(round(confidence * width))
    return f"[bright_blue]{'▮' * filled}[/][grey30]{'▯' * (width - filled)}[/]"


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
    beside it whenever the two lengths diverged — including the projected bars,
    which are part of the same picture and must be inside the same price range.
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


__all__ = [
    "confidence_meter",
    "format_bps",
    "format_price",
    "format_signed",
    "price_axis",
    "probability_bar",
    "projection_meter",
    "sparkline",
]
