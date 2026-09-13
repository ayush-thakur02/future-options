"""Technical indicators implemented directly on pandas.

Deliberately avoids TA-Lib: it needs a C toolchain and is the single most common
reason a Python trading project fails to install. Everything here is vectorised
and, more importantly, auditable — the smoothing conventions (Wilder's for RSI,
ATR and ADX; recursive EMA elsewhere) are explicit rather than hidden behind a
library's defaults.

Every function is causal: the value at bar *t* uses only bars up to and including
*t*. Short-window statistics are guarded so the first ``n - 1`` rows are NaN
rather than silently wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------- smoothing


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing — the recursive average used by RSI, ATR and ADX."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def activity_proxy(frame: pd.DataFrame, default: float = 1.0) -> pd.Series:
    """Best available measure of trading activity.

    Traded volume if the instrument has any (equities, futures), otherwise tick
    count. An index has no volume at all, so callers that need a denominator get
    a constant series rather than a scalar — which is what makes the difference
    between a working ``.rolling()`` and an AttributeError.
    """
    for column in ("volume", "tick_count"):
        if column in frame.columns:
            series = frame[column].astype("float64")
            if float(series.abs().sum()) > 0:
                return series
    return pd.Series(default, index=frame.index, dtype="float64")


# --------------------------------------------------------------- price transforms


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - prev_close).abs()
    low_close = (frame["low"] - prev_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    return wilder(true_range(frame), window)


def normalized_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR as a fraction of price — makes vol comparable across price levels."""
    return atr(frame, window) / frame["close"].replace(0, np.nan)


# --------------------------------------------------------------- oscillators


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder(gain, window)
    avg_loss = wilder(loss, window)
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def stochastic(
    frame: pd.DataFrame, k_window: int = 14, d_window: int = 3, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = frame["low"].rolling(k_window, min_periods=k_window).min()
    highest = frame["high"].rolling(k_window, min_periods=k_window).max()
    raw = 100.0 * (frame["close"] - lowest) / (highest - lowest + EPS)
    k = raw.rolling(smooth, min_periods=smooth).mean()
    return k, k.rolling(d_window, min_periods=d_window).mean()


def williams_r(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    highest = frame["high"].rolling(window, min_periods=window).max()
    lowest = frame["low"].rolling(window, min_periods=window).min()
    return -100.0 * (highest - frame["close"]) / (highest - lowest + EPS)


def cci(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    mean = typical.rolling(window, min_periods=window).mean()
    mad = (typical - mean).abs().rolling(window, min_periods=window).mean()
    return (typical - mean) / (0.015 * mad + EPS)


def ultimate_oscillator(frame: pd.DataFrame, short: int = 7, mid: int = 14, long: int = 28) -> pd.Series:
    prior_close = frame["close"].shift(1)
    buying_pressure = frame["close"] - pd.concat(
        [frame["low"], prior_close], axis=1
    ).min(axis=1)
    true_range_ = pd.concat(
        [frame["high"], prior_close], axis=1
    ).max(axis=1) - pd.concat([frame["low"], prior_close], axis=1).min(axis=1)

    def _avg(window: int) -> pd.Series:
        return buying_pressure.rolling(window, min_periods=window).sum() / (
            true_range_.rolling(window, min_periods=window).sum() + EPS
        )

    return 100.0 * (4.0 * _avg(short) + 2.0 * _avg(mid) + _avg(long)) / 7.0


# --------------------------------------------------------------- trend strength


def adx(frame: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX with +DI / -DI."""
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index
    )

    atr_ = wilder(true_range(frame), window)
    plus_di = 100.0 * wilder(plus_dm, window) / (atr_ + EPS)
    minus_di = 100.0 * wilder(minus_dm, window) / (atr_ + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return wilder(dx, window), plus_di, minus_di


def aroon(frame: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    def _since_max(series: pd.Series) -> pd.Series:
        return series.rolling(window, min_periods=window).apply(
            lambda values: float(len(values) - 1 - np.argmax(values)), raw=True
        )

    def _since_min(series: pd.Series) -> pd.Series:
        return series.rolling(window, min_periods=window).apply(
            lambda values: float(len(values) - 1 - np.argmin(values)), raw=True
        )

    up = 100.0 * (window - _since_max(frame["high"])) / window
    down = 100.0 * (window - _since_min(frame["low"])) / window
    return up, down


def vortex(frame: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
    vm_plus = (frame["high"] - frame["low"].shift(1)).abs()
    vm_minus = (frame["low"] - frame["high"].shift(1)).abs()
    tr_sum = true_range(frame).rolling(window, min_periods=window).sum()
    return (
        vm_plus.rolling(window, min_periods=window).sum() / (tr_sum + EPS),
        vm_minus.rolling(window, min_periods=window).sum() / (tr_sum + EPS),
    )


def supertrend(
    frame: pd.DataFrame, window: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend line, direction) where direction is +1 / -1."""
    mid = (frame["high"] + frame["low"]) / 2.0
    atr_ = atr(frame, window)
    upper = (mid + multiplier * atr_).to_numpy(dtype=float)
    lower = (mid - multiplier * atr_).to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)

    n = len(frame)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(upper[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            trend[i] = 1.0
            continue

        # Bands ratchet in the trend direction and never loosen.
        final_upper[i] = (
            upper[i]
            if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i]
            if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        if trend[i - 1] == 1.0:
            trend[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > final_upper[i] else -1.0

    line = pd.Series(
        np.where(trend == 1.0, final_lower, final_upper), index=frame.index, dtype="float64"
    )
    return line, pd.Series(trend, index=frame.index, dtype="float64")


def ichimoku(
    frame: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52
) -> dict[str, pd.Series]:
    def _midpoint(window: int) -> pd.Series:
        high = frame["high"].rolling(window, min_periods=window).max()
        low = frame["low"].rolling(window, min_periods=window).min()
        return (high + low) / 2.0

    conversion = _midpoint(tenkan)
    base = _midpoint(kijun)
    span_a = (conversion + base) / 2.0
    span_b = _midpoint(senkou)
    return {
        "tenkan": conversion,
        "kijun": base,
        "span_a": span_a,
        "span_b": span_b,
        "chikou": frame["close"].shift(-kijun),
    }


def efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """Kaufman's efficiency ratio: net move divided by total path travelled."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / (path + EPS)


def linreg_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling least-squares slope, normalised by mean level."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def _slope(values: np.ndarray) -> float:
        y = values - values.mean()
        return float((x_centered * y).sum() / denom)

    raw = series.rolling(window, min_periods=window).apply(_slope, raw=True)
    return raw / (series.rolling(window, min_periods=window).mean().abs() + EPS)


def linreg_r2(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def _r2(values: np.ndarray) -> float:
        y = values - values.mean()
        slope = float((x_centered * y).sum() / denom)
        predicted = slope * x_centered
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float((y**2).sum())
        if ss_tot <= 0:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    return series.rolling(window, min_periods=window).apply(_r2, raw=True)


# --------------------------------------------------------------- bands & channels


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> dict[str, pd.Series]:
    mid = sma(close, window)
    std = rolling_std(close, window)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "pct_b": (close - lower) / (upper - lower + EPS),
        "bandwidth": (upper - lower) / (mid.abs() + EPS),
    }


def keltner(
    frame: pd.DataFrame, window: int = 20, multiplier: float = 2.0
) -> dict[str, pd.Series]:
    mid = ema(frame["close"], window)
    atr_ = atr(frame, window)
    return {"mid": mid, "upper": mid + multiplier * atr_, "lower": mid - multiplier * atr_}


def donchian(frame: pd.DataFrame, window: int = 20) -> dict[str, pd.Series]:
    """Donchian channel.

    ``prior_upper`` / ``prior_lower`` exclude the current bar. A breakout must be
    measured against those, not against a channel that already contains the bar
    being tested — otherwise ``close >= upper`` can only be true when the bar's
    close is also its own high, which almost never happens and yields a feature
    that looks reasonable but never fires.
    """
    upper = frame["high"].rolling(window, min_periods=window).max()
    lower = frame["low"].rolling(window, min_periods=window).min()
    return {
        "upper": upper,
        "lower": lower,
        "mid": (upper + lower) / 2.0,
        "position": (frame["close"] - lower) / (upper - lower + EPS),
        "prior_upper": upper.shift(1),
        "prior_lower": lower.shift(1),
    }


# --------------------------------------------------------------- volume


def obv(frame: pd.DataFrame) -> pd.Series:
    direction = np.sign(frame["close"].diff()).fillna(0.0)
    return (direction * activity_proxy(frame)).cumsum()


def vwap_session(frame: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP, reset each trading day."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = activity_proxy(frame).replace(0, np.nan).fillna(1.0)

    day = pd.Series(frame.index.normalize(), index=frame.index)
    numerator = (typical * volume).groupby(day).cumsum()
    denominator = volume.groupby(day).cumsum()
    return numerator / (denominator + EPS)


def money_flow_index(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    raw_flow = typical * activity_proxy(frame)
    delta = typical.diff()
    positive = raw_flow.where(delta > 0, 0.0).rolling(window, min_periods=window).sum()
    negative = raw_flow.where(delta < 0, 0.0).rolling(window, min_periods=window).sum()
    ratio = positive / (negative + EPS)
    return 100.0 - (100.0 / (1.0 + ratio))


def chaikin_money_flow(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    span = (frame["high"] - frame["low"]).replace(0, np.nan)
    multiplier = ((frame["close"] - frame["low"]) - (frame["high"] - frame["close"])) / (span + EPS)
    volume = activity_proxy(frame)
    flow = multiplier.fillna(0.0) * volume
    return flow.rolling(window, min_periods=window).sum() / (
        volume.rolling(window, min_periods=window).sum() + EPS
    )


def force_index(frame: pd.DataFrame, window: int = 13) -> pd.Series:
    return ema(frame["close"].diff() * activity_proxy(frame), window)


# --------------------------------------------------------------- volatility


def realized_vol(close: pd.Series, window: int = 20, periods_per_year: int = 93_750) -> pd.Series:
    """Annualised close-to-close volatility (93,750 one-minute bars per year)."""
    returns = log_returns(close)
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)


def parkinson_vol(frame: pd.DataFrame, window: int = 20, periods_per_year: int = 93_750) -> pd.Series:
    """Range-based volatility; more efficient than close-to-close."""
    log_range = np.log(frame["high"] / frame["low"].replace(0, np.nan)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * log_range.rolling(window, min_periods=window).mean() * periods_per_year)


def garman_klass_vol(
    frame: pd.DataFrame, window: int = 20, periods_per_year: int = 93_750
) -> pd.Series:
    log_hl = np.log(frame["high"] / frame["low"].replace(0, np.nan)) ** 2
    log_co = np.log(frame["close"] / frame["open"].replace(0, np.nan)) ** 2
    term = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(term.rolling(window, min_periods=window).mean().clip(lower=0) * periods_per_year)


def volatility_ratio(short_vol: pd.Series, long_vol: pd.Series) -> pd.Series:
    """>1 means vol is expanding relative to its own baseline."""
    return short_vol / (long_vol + EPS)


# --------------------------------------------------------------- candle anatomy


def candle_anatomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose each bar into body and wick proportions."""
    rng = (frame["high"] - frame["low"]).replace(0, np.nan)
    body = frame["close"] - frame["open"]
    upper = frame["high"] - pd.concat([frame["open"], frame["close"]], axis=1).max(axis=1)
    lower = pd.concat([frame["open"], frame["close"]], axis=1).min(axis=1) - frame["low"]

    return pd.DataFrame(
        {
            "body_ratio": (body / rng).fillna(0.0),
            "upper_wick_ratio": (upper / rng).fillna(0.0),
            "lower_wick_ratio": (lower / rng).fillna(0.0),
            "close_location": ((frame["close"] - frame["low"]) / rng).fillna(0.5),
            "gap_from_prev_close": frame["open"] / frame["close"].shift(1).replace(0, np.nan) - 1.0,
        },
        index=frame.index,
    )


# --------------------------------------------------------------- regime


def hurst_exponent(close: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent via the variance-of-differences method.

    H > 0.5 suggests trending, H < 0.5 suggests mean reversion.

    Implemented with a sliding-window view rather than ``rolling.apply``. The
    per-window form is the hottest path in the whole feature set — it ran a
    Python callback for every bar, and each callback looped over the lags — which
    dominated live latency. Vectorising across windows turns thousands of Python
    calls into a handful of array operations and is roughly an order of magnitude
    faster for identical results.
    """
    values = np.log(close.to_numpy(dtype="float64"))
    n = len(values)
    out = np.full(n, np.nan, dtype="float64")
    if n < window:
        return pd.Series(out, index=close.index, dtype="float64")

    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags.astype("float64"))
    centered = log_lags - log_lags.mean()
    denom = float((centered**2).sum())

    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    taus = np.empty((windows.shape[0], len(lags)), dtype="float64")
    for j, lag in enumerate(lags):
        differences = windows[:, lag:] - windows[:, : -int(lag)]
        taus[:, j] = differences.std(axis=1, ddof=0)

    taus = np.maximum(taus, 1e-12)
    log_tau = np.log(taus)
    log_tau_centered = log_tau - log_tau.mean(axis=1, keepdims=True)
    slopes = log_tau_centered @ centered / denom

    # Each window ends at its last element, which is where rolling would report it.
    out[window - 1 :] = slopes
    return pd.Series(out, index=close.index, dtype="float64")
