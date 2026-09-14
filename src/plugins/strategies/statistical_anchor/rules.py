"""Beta-aware strategies relative to an optional benchmark or paired instrument.

The runtime can provide an aligned close series through ``StrategyContext.extras``
under ``anchor_close``, ``benchmark_close``, or as ``anchor_bars['close']``. A
single-instrument session has no defensible substitute, so these rules return a
neutral series until the anchor exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext, squash

EPS = 1e-12


def _anchor_close(context: StrategyContext) -> pd.Series | None:
    candidate = context.extras.get("anchor_close")
    if candidate is None:
        candidate = context.extras.get("benchmark_close")
    if candidate is None:
        candidate = context.extras.get("anchor_bars")
    if isinstance(candidate, pd.DataFrame):
        if "close" not in candidate.columns:
            return None
        candidate = candidate["close"]
    if not isinstance(candidate, pd.Series):
        return None
    return candidate.astype("float64").reindex(context.features.index).ffill()


def _asset_close(context: StrategyContext) -> pd.Series | None:
    if context.bars.empty or "close" not in context.bars.columns:
        return None
    return context.bars["close"].astype("float64").reindex(context.features.index)


def _rolling_beta(asset_return: pd.Series, anchor_return: pd.Series, window: int) -> pd.Series:
    covariance = asset_return.rolling(window, min_periods=window).cov(anchor_return)
    variance = anchor_return.rolling(window, min_periods=window).var(ddof=0)
    return (covariance / (variance + EPS)).clip(-5.0, 5.0)


class AnchorSpreadReversionStrategy(Strategy):
    name = "anchor_spread_reversion"
    category = "statistical"
    description = "Fade a standardized log-price spread after rolling beta neutralization"

    def __init__(self, window: int = 80, entry_z: float = 1.5) -> None:
        self.window = window
        self.entry_z = entry_z

    def score(self, context: StrategyContext) -> pd.Series:
        asset = _asset_close(context)
        anchor = _anchor_close(context)
        if asset is None or anchor is None:
            return self._empty(context)

        asset_log = np.log(asset.where(asset > 0))
        anchor_log = np.log(anchor.where(anchor > 0))
        beta = asset_log.rolling(self.window, min_periods=self.window).cov(anchor_log)
        beta /= anchor_log.rolling(self.window, min_periods=self.window).var(ddof=0) + EPS
        beta = beta.clip(-5.0, 5.0)

        intercept = asset_log.rolling(self.window, min_periods=self.window).mean()
        intercept -= beta * anchor_log.rolling(self.window, min_periods=self.window).mean()
        spread = asset_log - (intercept + beta * anchor_log)
        center = spread.rolling(self.window, min_periods=self.window).mean()
        scale = spread.rolling(self.window, min_periods=self.window).std(ddof=0)
        zscore = (spread - center) / (scale + EPS)

        asset_return = asset_log.diff()
        anchor_return = anchor_log.diff()
        correlation = asset_return.rolling(self.window, min_periods=self.window).corr(anchor_return)
        relationship = ((correlation.abs() - 0.20) / 0.55).clip(0.0, 1.0)
        excess = np.sign(zscore) * (zscore.abs() - self.entry_z).clip(lower=0.0)
        return squash(-excess * relationship, scale=1.0).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Beta-neutral spread stretched {'low' if value > 0 else 'high'}"


class RelativeStrengthRotationStrategy(Strategy):
    name = "relative_strength_rotation"
    category = "statistical"
    description = "Follow beta-adjusted relative strength versus an aligned anchor"

    def __init__(self, beta_window: int = 60, momentum_window: int = 20) -> None:
        self.beta_window = beta_window
        self.momentum_window = momentum_window

    def score(self, context: StrategyContext) -> pd.Series:
        asset = _asset_close(context)
        anchor = _anchor_close(context)
        if asset is None or anchor is None:
            return self._empty(context)

        asset_return = np.log(asset.where(asset > 0)).diff()
        anchor_return = np.log(anchor.where(anchor > 0)).diff()
        beta = _rolling_beta(asset_return, anchor_return, self.beta_window)
        residual = asset_return - beta * anchor_return
        relative_move = residual.rolling(
            self.momentum_window, min_periods=self.momentum_window
        ).sum()
        risk = residual.rolling(self.beta_window, min_periods=self.beta_window).std(ddof=0)
        risk_scaled = relative_move / (risk * np.sqrt(self.momentum_window) + EPS)

        correlation = asset_return.rolling(
            self.beta_window, min_periods=self.beta_window
        ).corr(anchor_return)
        relationship = ((correlation.abs() - 0.15) / 0.60).clip(0.0, 1.0)
        return squash(risk_scaled * relationship, scale=1.75).fillna(0.0).clip(-1.0, 1.0)

    def describe(self, value: float, context: StrategyContext) -> str:
        return f"Beta-adjusted relative strength {'outperforming' if value > 0 else 'lagging'}"


STRATEGIES = (
    AnchorSpreadReversionStrategy,
    RelativeStrengthRotationStrategy,
)
