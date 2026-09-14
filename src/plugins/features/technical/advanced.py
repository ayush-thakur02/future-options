"""Advanced causal indicators used by the technical feature plugin.

The functions in this module deliberately return continuous measurements rather
than buy/sell flags. Strategies and models can then choose their own thresholds,
while training and live inference share the exact same mathematics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import EPS, activity_proxy, ema, true_range


def kama(close: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's adaptive moving average."""
    change = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window, min_periods=window).sum()
    efficiency = (change / (volatility + EPS)).clip(0.0, 1.0)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    smoothing = (efficiency * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    values = close.to_numpy(dtype="float64")
    alpha = smoothing.to_numpy(dtype="float64")
    result = np.full(len(close), np.nan)
    if len(close) <= window:
        return pd.Series(result, index=close.index, dtype="float64")
    result[window] = float(np.nanmean(values[: window + 1]))
    for index in range(window + 1, len(values)):
        if np.isfinite(values[index]) and np.isfinite(alpha[index]):
            result[index] = result[index - 1] + alpha[index] * (
                values[index] - result[index - 1]
            )
        else:
            result[index] = result[index - 1]
    return pd.Series(result, index=close.index, dtype="float64")


def dema(close: pd.Series, window: int = 20) -> pd.Series:
    first = ema(close, window)
    second = ema(first, window)
    return 2.0 * first - second


def tema(close: pd.Series, window: int = 20) -> pd.Series:
    first = ema(close, window)
    second = ema(first, window)
    third = ema(second, window)
    return 3.0 * first - 3.0 * second + third


def trix(close: pd.Series, window: int = 15) -> pd.Series:
    triple = ema(ema(ema(close, window), window), window)
    return triple.pct_change(fill_method=None)


def percentage_price_oscillator(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    slow_average = ema(close, slow)
    line = (ema(close, fast) - slow_average) / (slow_average.abs() + EPS)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def fisher_transform(frame: pd.DataFrame, window: int = 10) -> tuple[pd.Series, pd.Series]:
    median = (frame["high"] + frame["low"]) / 2.0
    low = median.rolling(window, min_periods=window).min()
    high = median.rolling(window, min_periods=window).max()
    normalized = (2.0 * (median - low) / (high - low + EPS) - 1.0).clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    return fisher, fisher.shift(1)


def relative_vigor(frame: pd.DataFrame, window: int = 10) -> tuple[pd.Series, pd.Series]:
    numerator = (frame["close"] - frame["open"]).rolling(window, min_periods=window).mean()
    denominator = (frame["high"] - frame["low"]).rolling(window, min_periods=window).mean()
    vigor = numerator / (denominator + EPS)
    return vigor, vigor.rolling(4, min_periods=4).mean()


def choppiness(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    range_sum = true_range(frame).rolling(window, min_periods=window).sum()
    high = frame["high"].rolling(window, min_periods=window).max()
    low = frame["low"].rolling(window, min_periods=window).min()
    ratio = (range_sum / (high - low + EPS)).clip(lower=1.0)
    return 100.0 * np.log10(ratio) / np.log10(float(window))


def ulcer_index(close: pd.Series, window: int = 14) -> pd.Series:
    rolling_high = close.rolling(window, min_periods=window).max()
    drawdown_pct = 100.0 * (close / rolling_high.replace(0, np.nan) - 1.0)
    return np.sqrt(drawdown_pct.pow(2).rolling(window, min_periods=window).mean())


def downside_deviation(returns: pd.Series, window: int = 30) -> pd.Series:
    downside_squared = returns.clip(upper=0.0).pow(2)
    return np.sqrt(downside_squared.rolling(window, min_periods=window).mean())


def rolling_sortino(returns: pd.Series, window: int = 30) -> pd.Series:
    mean = returns.rolling(window, min_periods=window).mean()
    return mean / (downside_deviation(returns, window) + EPS)


def rolling_autocorrelation(
    returns: pd.Series, window: int = 50, lag: int = 1
) -> pd.Series:
    return returns.rolling(window, min_periods=window).corr(returns.shift(lag))


def variance_ratio(close: pd.Series, window: int = 60, lag: int = 5) -> pd.Series:
    one_period = np.log(close / close.shift(1))
    lagged = np.log(close / close.shift(lag))
    short_variance = one_period.rolling(window, min_periods=window).var(ddof=0)
    long_variance = lagged.rolling(window, min_periods=window).var(ddof=0)
    return long_variance / (lag * short_variance + EPS)


def directional_entropy(returns: pd.Series, window: int = 50) -> pd.Series:
    """Binary Shannon entropy of recent return signs, normalised to [0, 1]."""
    probability_up = (returns > 0).astype("float64").rolling(
        window, min_periods=window
    ).mean()
    probability_up = probability_up.clip(EPS, 1.0 - EPS)
    probability_down = 1.0 - probability_up
    return -(
        probability_up * np.log2(probability_up)
        + probability_down * np.log2(probability_down)
    )


def accumulation_distribution(frame: pd.DataFrame) -> pd.Series:
    activity = activity_proxy(frame)
    span = frame["high"] - frame["low"]
    multiplier = (
        (frame["close"] - frame["low"]) - (frame["high"] - frame["close"])
    ) / (span + EPS)
    return (multiplier.fillna(0.0) * activity.fillna(0.0)).cumsum()


def ease_of_movement(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    midpoint = (frame["high"] + frame["low"]) / 2.0
    distance = midpoint.diff()
    activity = activity_proxy(frame)
    box_ratio = activity / (frame["high"] - frame["low"] + EPS)
    raw = distance / (box_ratio + EPS)
    scale = raw.abs().rolling(window, min_periods=window).median()
    return (raw / (scale + EPS)).rolling(window, min_periods=window).mean()


def amihud_illiquidity(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = frame["close"].pct_change(fill_method=None).abs()
    activity = activity_proxy(frame)
    notional_proxy = frame["close"].abs() * activity.abs()
    return (returns / (notional_proxy + EPS)).rolling(window, min_periods=window).mean()


def mass_index(frame: pd.DataFrame, fast: int = 9, slow: int = 25) -> pd.Series:
    span = (frame["high"] - frame["low"]).abs()
    single = ema(span, fast)
    double = ema(single, fast)
    return (single / (double + EPS)).rolling(slow, min_periods=slow).sum()


__all__ = [
    "accumulation_distribution",
    "amihud_illiquidity",
    "choppiness",
    "dema",
    "directional_entropy",
    "downside_deviation",
    "ease_of_movement",
    "fisher_transform",
    "kama",
    "mass_index",
    "percentage_price_oscillator",
    "relative_vigor",
    "rolling_autocorrelation",
    "rolling_sortino",
    "tema",
    "trix",
    "ulcer_index",
    "variance_ratio",
]
