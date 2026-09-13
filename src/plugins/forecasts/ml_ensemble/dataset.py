"""Dataset construction: labelling and feature assembly.

The labelling choice here matters more than the model choice. Predicting the sign
of the next 5-minute return is close to unlearnable — the signal is tiny relative
to the noise, and a model that tries will spend its capacity fitting
microstructure noise that transaction costs would never let you trade.

So labels are **dead-banded**: a forward move smaller than a fraction of ATR is
dropped rather than forced into UP or DOWN. This concentrates the model on moves
that are actually big enough to matter and removes the ambiguous middle where
labelling is essentially arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.calendar import IST


@dataclass
class Dataset:
    """A labelled, horizon-specific training set."""

    X: pd.DataFrame
    y: pd.Series
    forward_return: pd.Series
    horizon: int
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def base_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0

    @property
    def index(self) -> pd.Index:
        return self.X.index

    def summary(self) -> dict:
        return {
            "horizon": self.horizon,
            "samples": len(self),
            "features": len(self.feature_names),
            "base_rate": round(self.base_rate, 4),
            "dropped_flat": int(self.forward_return.isna().sum()),
        }


def forward_returns(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Return from this bar's close to the close ``horizon`` bars ahead."""
    return bars["close"].shift(-horizon) / bars["close"] - 1.0


def session_mask(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Bars whose forward window stays inside the same trading day.

    A label that spans the overnight gap is not forecasting the same thing as one
    that does not — the close-to-open move is driven by news, global markets, and
    a different volatility regime. Mixing them in teaches the model to forecast
    something we never trade.
    """
    index = bars.index.tz_convert(IST) if bars.index.tz is not None else bars.index
    day = pd.Series(index.normalize(), index=bars.index)
    return day.eq(day.shift(-horizon))


def directional_labels(
    bars: pd.DataFrame,
    horizon: int,
    atr_norm: pd.Series,
    deadband: float = 0.12,
    hurdle_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Dead-banded binary direction labels.

    Returns ``(label, forward_return)`` where ``label`` is 1 for up, 0 for down,
    and NaN where the move was inside the dead band or crossed a session
    boundary. ``forward_return`` is retained in full so the backtest can compute
    realised PnL on the same rows — including the ones the model was not asked
    about.

    ``hurdle_bps`` raises the floor of the dead band to a move that could
    actually pay for itself. On a scalping timescale this is the difference
    between a usable model and an expensive one: the ATR-scaled band alone can
    sit well below the round-trip cost, so the model ends up being trained to
    classify moves no strategy could profit from. A prediction that is
    consistently right about a 0.3 bp move still loses money against a 4 bp
    round trip.
    """
    returns = forward_returns(bars, horizon)
    threshold = (deadband * atr_norm).abs()

    if hurdle_bps > 0:
        threshold = np.maximum(threshold, hurdle_bps / 10_000.0)

    valid = session_mask(bars, horizon)
    clear = returns.abs() > threshold

    label = pd.Series(np.nan, index=bars.index, dtype="float64")
    label[clear & valid & (returns > 0)] = 1.0
    label[clear & valid & (returns < 0)] = 0.0
    return label, returns


def build_dataset(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
    feature_names: list[str],
    deadband: float = 0.12,
    hurdle_bps: float = 0.0,
    dropna: bool = True,
) -> Dataset:
    """Assemble a labelled dataset for one forecast horizon."""
    atr_norm = features["atr_norm"] if "atr_norm" in features.columns else None
    if atr_norm is None:
        atr_norm = (features["atr"] if "atr" in features.columns else 0.001) * 0 + 0.001

    label, returns = directional_labels(
        bars, horizon, atr_norm, deadband=deadband, hurdle_bps=hurdle_bps
    )

    matrix = features[feature_names].copy()
    matrix["__label__"] = label
    matrix["__fwd__"] = returns
    if dropna:
        matrix = matrix.dropna(subset=["__label__"] + feature_names)
    else:
        matrix = matrix.dropna(subset=["__label__"])

    y = matrix.pop("__label__").astype(int)
    forward = matrix.pop("__fwd__")
    return Dataset(
        X=matrix,
        y=y,
        forward_return=forward,
        horizon=horizon,
        feature_names=feature_names,
    )


def sample_weights(dataset: Dataset) -> np.ndarray:
    """Weight samples inversely to label frequency.

    Without this, a dataset that is 54% UP produces a model that predicts UP
    almost always and looks accurate while being useless.
    """
    if len(dataset.y) == 0:
        return np.array([])
    counts = dataset.y.value_counts()
    total = counts.sum()
    weight_map = {label: total / (len(counts) * count) for label, count in counts.items()}
    return dataset.y.map(weight_map).to_numpy(dtype="float64")


def tradeable_fraction(
    bars: pd.DataFrame,
    horizon: int,
    hurdle_bps: float,
) -> dict:
    """How often a scalp over ``horizon`` could actually pay for itself.

    The first question to ask of any scalping setup, and the one most often
    skipped. If only a small share of bars produce a move larger than the
    round-trip cost, then no amount of model quality can make the timescale
    profitable — the ceiling is set by market structure, not by the model.
    """
    returns = forward_returns(bars, horizon)
    valid = session_mask(bars, horizon)
    live = returns[valid].dropna()
    if live.empty:
        return {"horizon": horizon, "bars": 0}

    threshold = hurdle_bps / 10_000.0
    clears = live.abs() > threshold

    return {
        "horizon": horizon,
        "bars": int(len(live)),
        "hurdle_bps": hurdle_bps,
        "tradeable_share": float(clears.mean()),
        "median_move_bps": float(live.abs().median() * 10_000),
        "p75_move_bps": float(live.abs().quantile(0.75) * 10_000),
        "p90_move_bps": float(live.abs().quantile(0.90) * 10_000),
        "up_share": float((live > 0).mean()),
    }


def class_balance(dataset: Dataset) -> dict:
    counts = dataset.y.value_counts().to_dict()
    return {
        "up": int(counts.get(1, 0)),
        "down": int(counts.get(0, 0)),
        "up_share": round(dataset.base_rate, 4),
    }
