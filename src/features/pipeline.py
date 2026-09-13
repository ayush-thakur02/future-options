"""Feature matrix assembly.

Produces a causal, model-ready frame from 1-minute OHLCV bars. Every column is
tagged with a group so the report can show where a model is actually drawing its
signal from, rather than treating it as one opaque blob.

Three deliberate choices:

* **Ichimoku's chikou span is excluded.** It is the close shifted backwards in
  time, so including it hands the model the future.
* **Intraday high/low references are running extremes**, not whole-session ones.
  A 09:30 bar must not know where the day's high ends up.
* **Volume-based indicators fall back to tick count.** An index has no
  consolidated tape, so `volume` is all zeros for NIFTY 50. Literal zeros would
  produce constant columns that look like features and carry nothing.

Columns are accumulated in a dict and concatenated once. Assigning 120 columns to
a DataFrame one at a time triggers pandas' fragmentation warning and is markedly
slower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .context import classify_regime, overnight_features, session_features

RETURN_LAGS = (1, 2, 3, 5, 10, 15, 30, 60)
VOL_WINDOWS = (10, 20, 60)

FEATURE_GROUPS: dict[str, list[str]] = {}
_Columns = dict[str, pd.Series]


def _register(group: str, columns: list[str]) -> None:
    bucket = FEATURE_GROUPS.setdefault(group, [])
    for column in columns:
        if column not in bucket:
            bucket.append(column)


def build_features(
    frame: pd.DataFrame,
    bar_minutes: int = 1,
    expiry_weekday: int = 1,
    include_context: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from OHLCV bars."""
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    data = frame.copy()
    close = data["close"].astype("float64")

    columns: _Columns = {}
    _add_momentum(columns, close)
    _add_trend(columns, close, data)
    _add_oscillators(columns, close, data)
    _add_volatility(columns, close, data)
    _add_activity(columns, close, data)
    _add_candle_shape(columns, data)
    _add_regime(columns, close)

    if include_context:
        context = pd.concat(
            [session_features(data, expiry_weekday=expiry_weekday), overnight_features(data)],
            axis=1,
        )
        for name in context.columns:
            columns[name] = context[name]
        _register("context", list(context.columns))

    features = pd.DataFrame(columns, index=frame.index)
    features["bar_minutes"] = float(bar_minutes)
    return features.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------- groups


def _add_momentum(out: _Columns, close: pd.Series) -> None:
    columns: list[str] = []

    for lag in RETURN_LAGS:
        name = f"ret_{lag}"
        out[name] = ta.log_returns(close, lag)
        columns.append(name)

    atr_norm = ta.normalized_atr(pd.DataFrame({"close": close, "high": close, "low": close}), 14)

    for window in (10, 20):
        name = f"ret_z_{window}"
        mean = out["ret_1"].rolling(window, min_periods=window).mean()
        std = out["ret_1"].rolling(window, min_periods=window).std(ddof=0)
        out[name] = (out["ret_1"] - mean) / (std + ta.EPS)
        columns.append(name)

    for window in (5, 10, 20, 60):
        name = f"roc_{window}"
        out[name] = close.pct_change(window)
        columns.append(name)

    direction = np.sign(out["ret_1"]).fillna(0.0)
    streak = direction.copy()
    for lag in range(1, 6):
        streak = streak + direction.shift(lag).fillna(0.0)
    out["direction_streak"] = streak
    columns.append("direction_streak")

    out["up_bar_ratio_20"] = (out["ret_1"] > 0).rolling(20, min_periods=20).mean()
    columns.append("up_bar_ratio_20")
    out["_atr_norm_ref"] = atr_norm

    _register("momentum", columns)


def _add_trend(out: _Columns, close: pd.Series, data: pd.DataFrame) -> None:
    columns: list[str] = []

    for window in (9, 21, 50, 200):
        ema_ = ta.ema(close, window)
        name = f"ema_dist_{window}"
        out[name] = close / ema_.replace(0, np.nan) - 1.0
        columns.append(name)

    out["ema_9_21_spread"] = out["ema_dist_9"] - out["ema_dist_21"]
    out["ema_21_50_spread"] = out["ema_dist_21"] - out["ema_dist_50"]
    out["ema_50_200_spread"] = out["ema_dist_50"] - out["ema_dist_200"]
    columns += ["ema_9_21_spread", "ema_21_50_spread", "ema_50_200_spread"]

    for fast, slow in ((9, 21), (21, 50), (50, 200)):
        name = f"ema_cross_{fast}_{slow}"
        out[name] = np.sign(out[f"ema_dist_{fast}"] - out[f"ema_dist_{slow}"])
        columns.append(name)

    adx_val, plus_di, minus_di = ta.adx(data, 14)
    out["adx_14"] = adx_val
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["di_spread"] = plus_di - minus_di
    columns += ["adx_14", "plus_di", "minus_di", "di_spread"]

    st_line, st_dir = ta.supertrend(data, 10, 3.0)
    out["supertrend_dir"] = st_dir
    out["supertrend_dist"] = close / st_line.replace(0, np.nan) - 1.0
    columns += ["supertrend_dir", "supertrend_dist"]

    aroon_up, aroon_down = ta.aroon(data, 25)
    out["aroon_up"] = aroon_up
    out["aroon_down"] = aroon_down
    out["aroon_osc"] = aroon_up - aroon_down
    columns += ["aroon_up", "aroon_down", "aroon_osc"]

    vortex_plus, vortex_minus = ta.vortex(data, 14)
    out["vortex_spread"] = vortex_plus - vortex_minus
    columns.append("vortex_spread")

    for window in (20, 50):
        name = f"slope_{window}"
        out[name] = ta.linreg_slope(close, window)
        columns.append(name)
    out["r2_20"] = ta.linreg_r2(close, 20)
    out["efficiency_ratio_10"] = ta.efficiency_ratio(close, 10)
    columns += ["r2_20", "efficiency_ratio_10"]

    ichimoku = ta.ichimoku(data)
    # chikou is intentionally omitted: it is the close shifted forward in time.
    for name in ("tenkan", "kijun", "span_a", "span_b"):
        column = f"ichimoku_{name}_dist"
        out[column] = close / ichimoku[name].replace(0, np.nan) - 1.0
        columns.append(column)
    out["ichimoku_cloud_top"] = np.maximum(
        out["ichimoku_span_a_dist"], out["ichimoku_span_b_dist"]
    )
    out["ichimoku_cloud_bottom"] = np.minimum(
        out["ichimoku_span_a_dist"], out["ichimoku_span_b_dist"]
    )
    columns += ["ichimoku_cloud_top", "ichimoku_cloud_bottom"]

    _register("trend", columns)


def _add_oscillators(out: _Columns, close: pd.Series, data: pd.DataFrame) -> None:
    columns: list[str] = []

    for window in (7, 14, 21):
        name = f"rsi_{window}"
        out[name] = ta.rsi(close, window)
        columns.append(name)
    out["rsi_14_slope"] = out["rsi_14"].diff(3)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(float)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(float)
    columns += ["rsi_14_slope", "rsi_overbought", "rsi_oversold"]

    macd_line, macd_signal, macd_hist = ta.macd(close)
    safe_close = close.replace(0, np.nan)
    out["macd_line"] = macd_line / safe_close
    out["macd_signal"] = macd_signal / safe_close
    out["macd_hist"] = macd_hist / safe_close
    out["macd_cross"] = np.sign(macd_hist)
    columns += ["macd_line", "macd_signal", "macd_hist", "macd_cross"]

    stoch_k, stoch_d = ta.stochastic(data)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["stoch_cross"] = stoch_k - stoch_d
    columns += ["stoch_k", "stoch_d", "stoch_cross"]

    out["williams_r"] = ta.williams_r(data, 14)
    out["cci_20"] = ta.cci(data, 20)
    out["ultimate_osc"] = ta.ultimate_oscillator(data)
    out["mfi_14"] = ta.money_flow_index(data, 14)
    out["cmf_20"] = ta.chaikin_money_flow(data, 20)
    columns += ["williams_r", "cci_20", "ultimate_osc", "mfi_14", "cmf_20"]

    _register("oscillator", columns)


def _add_volatility(out: _Columns, close: pd.Series, data: pd.DataFrame) -> None:
    columns: list[str] = []

    atr_norm = ta.normalized_atr(data, 14)
    out["atr_norm"] = atr_norm
    out["atr_ratio"] = atr_norm / (atr_norm.rolling(50, min_periods=50).mean() + ta.EPS)
    columns += ["atr_norm", "atr_ratio"]

    bollinger = ta.bollinger(close, 20, 2.0)
    out["bb_pct_b"] = bollinger["pct_b"]
    out["bb_bandwidth"] = bollinger["bandwidth"]
    out["bb_squeeze"] = bollinger["bandwidth"] / (
        bollinger["bandwidth"].rolling(100, min_periods=50).mean() + ta.EPS
    )
    columns += ["bb_pct_b", "bb_bandwidth", "bb_squeeze"]

    keltner = ta.keltner(data, 20, 2.0)
    keltner_span = (keltner["upper"] - keltner["lower"]).replace(0, np.nan)
    out["keltner_position"] = (close - keltner["lower"]) / keltner_span
    columns.append("keltner_position")

    donchian = ta.donchian(data, 20)
    out["donchian_position"] = donchian["position"]
    out["donchian_width"] = (donchian["upper"] - donchian["lower"]) / close.replace(0, np.nan)
    # Breakouts are judged against the channel as it stood *before* this bar.
    out["breakout_high_20"] = (close > donchian["prior_upper"]).astype(float)
    out["breakout_low_20"] = (close < donchian["prior_lower"]).astype(float)
    columns += ["donchian_position", "donchian_width", "breakout_high_20", "breakout_low_20"]

    for window in VOL_WINDOWS:
        name = f"realized_vol_{window}"
        out[name] = ta.realized_vol(close, window)
        columns.append(name)

    out["parkinson_vol_20"] = ta.parkinson_vol(data, 20)
    out["garman_klass_vol_20"] = ta.garman_klass_vol(data, 20)
    out["vol_ratio_10_60"] = ta.volatility_ratio(out["realized_vol_10"], out["realized_vol_60"])
    out["vol_ratio_20_60"] = ta.volatility_ratio(out["realized_vol_20"], out["realized_vol_60"])
    columns += [
        "parkinson_vol_20",
        "garman_klass_vol_20",
        "vol_ratio_10_60",
        "vol_ratio_20_60",
    ]

    out["return_skew_30"] = out["ret_1"].rolling(30, min_periods=30).skew()
    out["return_kurt_30"] = out["ret_1"].rolling(30, min_periods=30).kurt()
    columns += ["return_skew_30", "return_kurt_30"]

    _register("volatility", columns)


def _add_activity(out: _Columns, close: pd.Series, data: pd.DataFrame) -> None:
    """Volume features, with tick count as the activity proxy for indices."""
    columns: list[str] = []

    has_volume = "volume" in data.columns and float(data["volume"].abs().sum()) > 0
    has_ticks = "tick_count" in data.columns and float(data["tick_count"].abs().sum()) > 0

    activity = ta.activity_proxy(data, default=1.0)
    mean_20 = activity.rolling(20, min_periods=20).mean()
    std_20 = activity.rolling(20, min_periods=20).std(ddof=0)

    out["activity_ratio_20"] = activity / (mean_20 + ta.EPS)
    out["activity_z_20"] = (activity - mean_20) / (std_20 + ta.EPS)
    out["activity_slope_20"] = ta.linreg_slope(
        activity.replace(0, np.nan).ffill().fillna(0.0), 20
    )
    columns += ["activity_ratio_20", "activity_z_20", "activity_slope_20"]

    vwap = ta.vwap_session(data)
    out["vwap_dist"] = close / vwap.replace(0, np.nan) - 1.0
    out["vwap_above"] = (close > vwap).astype(float)
    columns += ["vwap_dist", "vwap_above"]

    obv = ta.obv(data)
    out["obv_slope_20"] = ta.linreg_slope(obv.replace(0, np.nan).ffill(), 20)
    out["obv_dist"] = obv / (obv.rolling(50, min_periods=50).mean().abs() + ta.EPS) - 1.0
    columns += ["obv_slope_20", "obv_dist"]

    if has_volume:
        _register("activity", columns)
    elif has_ticks:
        _register("activity(tick-proxy)", columns)
    else:
        # Neither real volume nor ticks: these collapse to constants and will be
        # dropped by feature_columns(). Record that honestly rather than pretend.
        _register("activity(unavailable)", columns)


def _add_candle_shape(out: _Columns, data: pd.DataFrame) -> None:
    anatomy = ta.candle_anatomy(data)
    for column in anatomy.columns:
        out[column] = anatomy[column]
    _register("candle", list(anatomy.columns))


def _add_regime(out: _Columns, close: pd.Series) -> None:
    columns: list[str] = []

    out["hurst_100"] = ta.hurst_exponent(close, 100)
    mean_100 = close.rolling(100, min_periods=100).mean()
    std_100 = close.rolling(100, min_periods=100).std(ddof=0)
    out["price_z_100"] = (close - mean_100) / (std_100 + ta.EPS)
    columns += ["hurst_100", "price_z_100"]

    _register("regime", columns)


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model input columns: numeric and non-constant."""
    excluded = {"bar_minutes", "_atr_norm_ref"}
    columns: list[str] = []
    for column in matrix.columns:
        if column in excluded or column.startswith("_"):
            continue
        if not pd.api.types.is_numeric_dtype(matrix[column]):
            continue
        if matrix[column].nunique(dropna=True) <= 1:
            continue
        columns.append(column)
    return columns


def regime_labels(matrix: pd.DataFrame, window: int = 100) -> pd.Series:
    return classify_regime(matrix, atr_col="atr_norm", window=window)
